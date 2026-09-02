import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 684
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_ln_ln_softmax_gelu(
    X, G0, B0, G1, B1, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, round to bf16 like PyTorch) ----
    mean0 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(xc * xc, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + EPS)

    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd0 * g0 + b0).to(tl.bfloat16).to(tl.float32)
    y = tl.where(mask, y, 0.0)

    # ---- LayerNorm 1 ----
    mean1 = tl.sum(y, axis=0) / N
    yc = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(yc * yc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (yc * rstd1 * g1 + b1).to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 math, round to bf16) ----
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact erf form, fp32 opmath like PyTorch CUDA) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.softmax(y, dim=-1)
            return F.gelu(y)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        M_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_ln_softmax_gelu[(M_rows,)](
            x2, self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b, out,
            N, x2.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
