import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 593
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B2, W, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (compute in fp32, round to fp16 like PyTorch intermediate storage)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # gelu #2
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # bias add
    b = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    x = x.to(tl.float16).to(tl.float32)

    # RMS norm (in fp32)
    ms = tl.sum(x * x, axis=0) / D_
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (x * inv).to(tl.float16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.float16)

    # relu
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * D_ + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.b2, self.rms3_w, y,
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
