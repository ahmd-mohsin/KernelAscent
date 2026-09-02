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
    stride_x, stride_y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32 (matches PyTorch bf16 gelu which uses fp32 opmath)
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    # cast to bf16 (output dtype of F.gelu) then back to fp32 like reference
    g_bf16 = g.to(tl.bfloat16)
    xf = g_bf16.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)

    normed = (xf * r).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = normed * w

    tl.store(Y + row * stride_y + cols, out, mask=mask)


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
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _gelu_rms_kernel[(m,)](
            x2, self.rms1_w, y,
            x2.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
