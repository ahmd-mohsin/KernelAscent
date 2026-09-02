import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 811
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_bias_softmax(X, B0, B1, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)

    # match bf16 rounding of the two separate adds
    x = (x + b0).to(tl.bfloat16)
    x = (x + b1).to(tl.bfloat16)

    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_bias_softmax[(m,)](
            x, self.b0, self.b1, y, d, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return y
