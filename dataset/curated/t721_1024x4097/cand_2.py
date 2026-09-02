import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 721
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_ln_add_gelu_ln(
    X, G1, B1, B2, G4, B4, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    n_f = N.to(tl.float32)

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, erf-based), round to bf16 like the eager op ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 (fp32 accumulation) ----
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)

    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g1 + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- add bias b2, round to bf16 ----
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b2
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- GELU 2 ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    xm = tl.where(mask, x, 0.0)
    mean2 = tl.sum(xm, axis=0) / n_f
    d2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)

    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g4 + b4

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x + self.b2
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_add_gelu_ln[(rows,)](
            x2, self.ln1_g, self.ln1_b, self.b2, self.ln4_g, self.ln4_b, y,
            N, x2.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
