import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 201
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact, erf-based) computed in fp32, then cast to bf16 (matches F.gelu on bf16)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16)

    # relu
    r = tl.maximum(g, 0.0)

    # rmsnorm in fp32
    rf = r.to(tl.float32)
    ms = tl.sum(rf * rf, axis=0) / D_
    inv = tl.math.rsqrt(ms + 1e-6)
    n = (rf * inv).to(tl.bfloat16)

    # multiply weight in bf16
    w = tl.load(W + offs, mask=mask, other=0.0)
    h = n * w

    # final gelu: fp32 compute, cast back
    hf = h.to(tl.float32)
    out = hf * 0.5 * (1.0 + tl.math.erf(hf * INV_SQRT2))
    out = out.to(tl.bfloat16)

    tl.store(Y + row * D_ + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        _fused_kernel[(m,)](
            x, self.rms2_w, y,
            D_=d, BLOCK=triton.next_power_of_2(d),
            num_warps=8,
        )
        return y
