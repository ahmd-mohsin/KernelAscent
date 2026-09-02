import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 723
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_ln_ln_gelu_gelu(
    X, Y, G1, B1, G2, B2,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 accumulation, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g1 + b1
    # round to fp16 as the reference does between ops
    y = y.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    yc = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = yc * rstd2 * g2 + b2
    z = z.to(tl.float16).to(tl.float32)

    # GELU (exact, erf-based) twice with fp16 rounding in between
    inv_sqrt2 = 0.7071067811865476
    z = 0.5 * z * (1.0 + tl.math.erf(z * inv_sqrt2))
    z = z.to(tl.float16).to(tl.float32)
    z = 0.5 * z * (1.0 + tl.math.erf(z * inv_sqrt2))

    tl.store(Y + row * N + cols, z.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_gelu_gelu[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
