import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 401
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _bias_triple_ln_kernel(
    X, B2,
    G3, B3, G4, B4, G5, B5,
    Y,
    N, EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row * N

    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    # bf16 add (compute in fp32, round to bf16 to match torch bf16 elementwise add)
    xb = (x + b2).to(tl.bfloat16)
    xf = xb.to(tl.float32)

    # ---- LayerNorm 3 ----
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g + b).to(tl.bfloat16)  # round to bf16 (matches torch inter-op cast)
    xf = y.to(tl.float32)

    # ---- LayerNorm 4 ----
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g + b).to(tl.bfloat16)
    xf = y.to(tl.float32)

    # ---- LayerNorm 5 ----
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G5 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B5 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g + b).to(tl.bfloat16)

    tl.store(Y + base + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmuls (already optimal); keep two-step to preserve numerics
        h = x @ self.W0
        h = h @ self.W1
        h = h.contiguous()

        rows, N = h.shape[0], h.shape[-1]
        h2d = h.view(-1, N)
        rows = h2d.shape[0]
        out = torch.empty_like(h2d)

        BLOCK = triton.next_power_of_2(N)
        _bias_triple_ln_kernel[(rows,)](
            h2d, self.b2,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            self.ln5_g, self.ln5_b,
            out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view_as(h)
