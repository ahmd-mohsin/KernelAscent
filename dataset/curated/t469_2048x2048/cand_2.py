import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 469
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B2, G3, Bt3, G4, Bt4, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 internal, output rounded to bf16) ----
    row_max = tl.max(x, axis=0)
    ex = tl.exp(x - row_max)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    s = ex / denom
    s = s.to(tl.bfloat16).to(tl.float32)

    # ---- gelu (erf-based, fp32 math, rounded to bf16) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = s * 0.5 * (1.0 + tl.math.erf(s * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ---- add bias (bf16 rounding) ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    a = (g + b2).to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 3 ----
    Nf = N * 1.0
    mean1 = tl.sum(tl.where(mask, a, 0.0), axis=0) / Nf
    d1 = tl.where(mask, a - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / Nf
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(Bt3 + cols, mask=mask, other=0.0).to(tl.float32)
    h = (d1 * rstd1 * g3 + b3).to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 4 ----
    mean2 = tl.sum(tl.where(mask, h, 0.0), axis=0) / Nf
    d2 = tl.where(mask, h - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / Nf
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(Bt4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (d2 * rstd2 * g4 + b4).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x2, self.b2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, y,
            x2.stride(0), y.stride(0),
            N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
