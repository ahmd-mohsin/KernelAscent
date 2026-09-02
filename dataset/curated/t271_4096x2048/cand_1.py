import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 271
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B, W, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact), round to fp16 like PyTorch
    inv_sqrt2 = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    x = x.to(tl.float16).to(tl.float32)

    # add bias, round to fp16
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.float16).to(tl.float32)

    # gelu again, round to fp16
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    x = x.to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D_
    r = tl.math.rsqrt(ms + 1e-6)
    xn = (x * r).to(tl.float16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.float16)
    tl.store(Y + row * D_ + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.b1, self.rms3_w, y,
            Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
