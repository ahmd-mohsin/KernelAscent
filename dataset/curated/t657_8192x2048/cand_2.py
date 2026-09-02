import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 657
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, W0, W1, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 0 (fp32 compute, round to bf16, then bf16-weighted mul rounded to bf16)
    ms = tl.sum(x * x, axis=0) / D
    rs = 1.0 / tl.sqrt(ms + 1e-6)
    y = (x * rs).to(tl.bfloat16).to(tl.float32)
    x1 = (y * w0).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 1
    ms1 = tl.sum(tl.where(mask, x1 * x1, 0.0), axis=0) / D
    rs1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    y1 = (x1 * rs1).to(tl.bfloat16).to(tl.float32)
    x2 = (y1 * w1).to(tl.bfloat16).to(tl.float32)

    # GELU (erf) twice, rounding to bf16 between (matches PyTorch bf16 elementwise)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = 0.5 * x2 * (1.0 + tl.math.erf(x2 * INV_SQRT2))
    g1 = g1.to(tl.bfloat16).to(tl.float32)
    g2 = 0.5 * g1 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.bfloat16)

    tl.store(OUT + row * D + offs, g2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.rms0_w, self.rms1_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
