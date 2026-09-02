import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 647
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_rms_ln_softmax_gelu(
    X, W1, G2, B2, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    base = row * N

    # ---- load row (bf16 -> f32) ----
    x = tl.load(X + base + cols).to(tl.float32)

    # ---- RMSNorm (computed in f32, cast to bf16, then bf16 scale) ----
    ms = tl.sum(x * x, axis=0) / N
    xr = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16)
    w1 = tl.load(W1 + cols)  # bf16
    x1 = xr * w1             # bf16 multiply (matches PyTorch rounding)

    # ---- LayerNorm (f32 accumulation, eps=1e-5) ----
    xf = x1.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = xf - mean
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G2 + cols).to(tl.float32)
    b = tl.load(B2 + cols).to(tl.float32)
    x2 = (d * inv * g + b).to(tl.bfloat16)

    # ---- Softmax (f32) ----
    xf2 = x2.to(tl.float32)
    mx = tl.max(xf2, axis=0)
    e = tl.exp(xf2 - mx)
    s = tl.sum(e, axis=0)
    x3 = (e / s).to(tl.bfloat16)

    # ---- GELU (erf-based, f32) ----
    xf3 = x3.to(tl.float32)
    y = 0.5 * xf3 * (1.0 + tl.math.erf(xf3 * 0.7071067811865476))

    tl.store(Y + base + cols, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_rms_ln_softmax_gelu[(Mrows,)](
            h, self.rms1_w, self.ln2_g, self.ln2_b, out,
            N=N, BLOCK=N,
            num_warps=8,
        )
        return out
