import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 75
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_rms_gelu_ln_ln(
    X, RMSW, G3, B3, G4, B4, Out,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N + offs

    x = tl.load(X + base).to(tl.float32)

    # ---- RMSNorm (eps=1e-6), computed in fp32, cast to bf16, then * weight ----
    ms = tl.sum(x * x, 0) / N
    xr = x * tl.math.rsqrt(ms + 1e-6)
    xr = xr.to(tl.bfloat16).to(tl.float32)
    w = tl.load(RMSW + offs).to(tl.float32)
    x = (xr * w).to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf), fp32 opmath, cast back to bf16 ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 (eps=1e-5) ----
    mean = tl.sum(x, 0) / N
    xm = x - mean
    var = tl.sum(xm * xm, 0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g3 = tl.load(G3 + offs).to(tl.float32)
    b3 = tl.load(B3 + offs).to(tl.float32)
    x = xm * rstd * g3 + b3
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 4 (eps=1e-5) ----
    mean = tl.sum(x, 0) / N
    xm = x - mean
    var = tl.sum(xm * xm, 0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g4 = tl.load(G4 + offs).to(tl.float32)
    b4 = tl.load(B4 + offs).to(tl.float32)
    y = xm * rstd * g4 + b4

    tl.store(Out + base, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores (already optimal on A100)
        x = torch.matmul(x, self.W0)
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        # Single fused kernel: RMSNorm -> GELU -> LayerNorm -> LayerNorm
        # (one read + one write of the activation instead of 8+ memory passes)
        _fused_rms_gelu_ln_ln[(Mrows,)](
            x, self.rms1_w, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            N=N, BLOCK=N,
            num_warps=8,
        )
        return out
