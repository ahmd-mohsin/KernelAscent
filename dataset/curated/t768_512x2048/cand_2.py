import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 768
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, stride_x, stride_y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (float accumulation, output rounded to fp16 like PyTorch)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = (e / denom).to(tl.float16).to(tl.float32)

    # exact GELU (erf), computed in fp32, rounded to fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * s * (1.0 + tl.math.erf(s * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # scalar multiply, rounded to fp16
    g = (g * 1.1037).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    n = (g * r).to(tl.float16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (n * w).to(tl.float16).to(tl.float32)

    # relu
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x, self.rms3_w, y,
            x.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
