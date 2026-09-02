import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 900
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_bias_gelu_softmax(X, B, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    x = x + b
    # exact GELU (erf), computed in fp32 like PyTorch's opmath for half
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # match PyTorch: gelu output is stored as fp16 before softmax
    g = g.to(tl.float16).to(tl.float32)

    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_bias_gelu_softmax[(m,)](
            x, self.b0, y, d, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return y
