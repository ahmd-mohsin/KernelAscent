import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 788
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _ln3_gelu_kernel(
    X, Y,
    G1, B1, G2, B2, G3, B3,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    n = N.to(tl.float32)

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LN1
    mean = tl.sum(x, axis=0) / n
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xc * rstd) * g + b
    x = x.to(tl.float16).to(tl.float32)  # match fp16 round-trip between ops

    # LN2
    mean = tl.sum(x, axis=0) / n
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xc * rstd) * g + b
    x = x.to(tl.float16).to(tl.float32)

    # LN3
    mean = tl.sum(x, axis=0) / n
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xc * rstd) * g + b
    x = x.to(tl.float16).to(tl.float32)

    # GELU (exact, erf-based)
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    y = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))

    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln3_gelu_kernel[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
