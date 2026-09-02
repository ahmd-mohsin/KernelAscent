import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 205
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W, B, B4, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)

    # relu (exact in bf16)
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # gelu #1 (exact erf), round to bf16 to match PyTorch intermediate storage
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g1 = g1.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g2 = g1 * 0.5 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # layer norm in fp32
    g2m = tl.where(mask, g2, 0.0)
    mean = tl.sum(g2m, axis=0) / N
    diff = tl.where(mask, g2 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g2 - mean) * rstd * w + b
    y = y.to(tl.bfloat16)

    # add bias b4 in bf16
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    y = y + b4

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x, self.ln3_g, self.ln3_b, self.b4, y,
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
