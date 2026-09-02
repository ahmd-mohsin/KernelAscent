import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 467
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_kernel(X, G, B, Y, N, eps, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    # relu (fp16, exact)
    x = tl.maximum(x, 0.0)
    # gelu in fp32 opmath, round to fp16 (mimics separate op storing fp16)
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16)

    # layernorm with fp32 accumulation
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)
    mean = tl.sum(gf, axis=0) / N
    diff = tl.where(mask, gf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (gf - mean) * rstd * w + b
    y = y.to(tl.float16)

    # relu + scale (fp32 opmath, round fp16)
    y = tl.maximum(y, 0.0)
    y = (y.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.ln2_g, self.ln2_b, y,
            N, 1e-5, 1.2041,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
