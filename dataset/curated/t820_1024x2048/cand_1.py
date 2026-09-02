import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 820
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _gelu_rms_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round to bf16 like F.gelu output, then upcast (matches _xf = x.float())
    g_bf = g.to(tl.bfloat16)
    xf = g_bf.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    # (xf * inv).to(bf16) * w  (bf16 elementwise mul == fp32 mul + round)
    normed = (xf * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (normed * w).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _gelu_rms_kernel[(m,)](
            x2, self.rms1_w, y,
            x2.stride(0), y.stride(0),
            n, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
