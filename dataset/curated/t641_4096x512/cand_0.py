import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 641
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, Out, stride_x, stride_o, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (compute in fp32, round to bf16, scale by weight in fp32-opmath, round to bf16)
    ms = tl.sum(x * x, axis=0) / D_
    rs = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * rs).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)

    # GELU (exact, erf) twice - fp32 opmath, round to bf16 between ops
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    g1 = g1.to(tl.bfloat16).to(tl.float32)
    g2 = 0.5 * g1 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32
    g2m = tl.where(mask, g2, float('-inf'))
    mx = tl.max(g2m, axis=0)
    e = tl.exp(g2m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.rms0_w, out,
            x.stride(0), out.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
