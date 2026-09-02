import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 995
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_ln_gelu2_rms_relu(
    X, G, B, W, Y,
    N,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (stats in fp32, like PyTorch half layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b
    y = y.to(tl.float16).to(tl.float32)  # round-trip like real fp16 output

    # ---- GELU (exact erf), fp32 opmath, rounded to fp16 between ops ----
    SQ: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQ))
    y = y.to(tl.float16).to(tl.float32)
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQ))
    y = y.to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32) ----
    yy = tl.where(mask, y * y, 0.0)
    ms = tl.sum(yy, axis=0) / N
    r = tl.math.rsqrt(ms + EPS_RMS)
    y = (y * r).to(tl.float16).to(tl.float32)

    # ---- scale by rms weight (fp16 mul == exact fp32 product rounded to fp16) ----
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.float16)

    # ---- ReLU ----
    zero = tl.zeros(out.shape, dtype=tl.float16)
    out = tl.maximum(out, zero)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_gelu2_rms_relu[(Mrows,)](
            x, self.ln1_g, self.ln1_b, self.rms4_w, out,
            N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
