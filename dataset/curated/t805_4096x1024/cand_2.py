import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 805
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_ln_relu_gelu_ln(
    X, Y, G1, B1, G4, B4,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1
    # round to fp16 (matches PyTorch intermediate tensor dtype)
    y1 = y1.to(tl.float16).to(tl.float32)

    # ---- ReLU ----
    y1 = tl.maximum(y1, 0.0)

    # ---- GELU (exact, erf) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865475
    y2 = y1 * 0.5 * (1.0 + tl.math.erf(y1 * INV_SQRT2))
    y2 = y2.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(tl.where(mask, y2, 0.0), axis=0) / N
    d2 = tl.where(mask, y2 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)

    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g4 + b4

    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_relu_gelu_ln[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.ln4_g, self.ln4_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
