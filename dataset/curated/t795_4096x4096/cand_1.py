import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 795
M, D, DT = 4096, 4096, torch.bfloat16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_kernel(X, B, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact), round back to bf16 like the eager op
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # add bias, round
    g = (g + b)
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale, round
    g = g * 1.4866
    g = g.to(tl.bfloat16).to(tl.float32)

    # gelu again, round
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch acc_type behavior)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * D + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape[0] * (x.numel() // (x.shape[-1] * x.shape[0]) if x.dim() > 2 else 1), x.shape[-1]
        x2 = x.view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](x2, self.b1, y, d, BLOCK=BLOCK, num_warps=num_warps)
        return y.view(x.shape)
