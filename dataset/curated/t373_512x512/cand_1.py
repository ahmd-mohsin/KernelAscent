import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 373
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_ln_gelu(X, G, B, Y, N, eps,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), computed in fp32 then rounded to fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(g1, axis=0) / N
    diff = tl.where(mask, g1 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * gamma + beta
    y = y.to(tl.float16).to(tl.float32)

    # second GELU
    g2 = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    tl.store(Y + row * N + cols, g2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _fused_gelu_ln_gelu[(Mrows,)](
            h, self.ln2_g, self.ln2_b, y,
            N, 1e-5,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y
