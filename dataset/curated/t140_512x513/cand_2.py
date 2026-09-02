import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 140
M, D, DT = 512, 513, torch.float16


@triton.jit
def _fused_gelu2_bias_softmax(X, B, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # First GELU (exact, erf-based), computed in fp32 then rounded to fp16
    # to match PyTorch's half-precision op-by-op behavior.
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Second GELU
    g = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Bias add (fp32 add of two fp16 values is exact; round to fp16 matches
    # correctly-rounded half addition)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g + b)
    y = y.to(tl.float16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch half softmax which accumulates in float)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu2_bias_softmax[(rows,)](
            h, self.b3, out, N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
