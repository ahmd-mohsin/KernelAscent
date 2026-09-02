import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 259
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_bias_rms_gelu(
    X, B, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # x = x + b0 (bf16 add == fp32 add then round to bf16)
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = xb.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    rs = tl.math.rsqrt(ms + eps)

    # cast normalized value to bf16, then multiply by weight (bf16 mult)
    xn = (xf * rs).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    v16 = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # exact GELU computed in fp32 (matches PyTorch opmath for bf16)
    v = v16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * v * (1.0 + tl.math.erf(v * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xx = x + self.b0
            _xf = xx.float()
            xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms1_w
            return F.gelu(xx)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_rms_gelu[(m,)](
            x2, self.b0, self.rms1_w, y,
            n, x2.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
