import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 992
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # relu + bias (exact in fp16 semantics: inputs are exact fp16, fp32 add then round)
    x = tl.maximum(x, 0.0)
    x = x + b
    x = x.to(tl.float16).to(tl.float32)

    # exact (erf-based) GELU in fp32, matching CUDA opmath behavior
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g16 = g.to(tl.float16)

    # RMSNorm in fp32 on the fp16-rounded values
    gf = g16.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    out = (gf * inv).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    out = out * w
    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](x, self.b2, self.rms4_w, y, d, BLOCK=BLOCK,
                            num_warps=8)
        return y
