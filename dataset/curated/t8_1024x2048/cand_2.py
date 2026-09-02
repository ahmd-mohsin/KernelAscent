import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 8
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_gelu_softmax_bias(X, B, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32 (matches PyTorch opmath), then round to fp16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32 accumulation (matches PyTorch accscalar)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # bias add in fp16 (matches half+half elementwise add)
    b = tl.load(B + offs, mask=mask, other=0.0)
    y = p + b
    tl.store(Y + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_gelu_softmax_bias[(rows,)](
            x, self.b2, y, d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
