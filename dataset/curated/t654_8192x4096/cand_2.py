import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 654
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_norm_act_kernel(
    X, W_RMS, LN_G, LN_B, OUT,
    N, stride_x, stride_o,
    EPS_RMS: tl.constexpr, EPS_LN: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # ---- RMSNorm (compute in fp32, round to fp16, multiply by weight in fp16) ----
    ms = tl.sum(xf * xf, axis=0) / N
    rstd_rms = 1.0 / tl.sqrt(ms + EPS_RMS)
    y_h = (xf * rstd_rms).to(tl.float16)
    w = tl.load(W_RMS + cols, mask=mask, other=0.0)  # fp16
    y_h = y_h * w  # fp16 multiply (matches PyTorch half*half)

    # ---- LayerNorm (opmath fp32, output rounded to fp16) ----
    yf = y_h.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mu = tl.sum(yf, axis=0) / N
    diff = tl.where(mask, yf - mu, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    z = (yf - mu) * rstd * g + b
    z_h = z.to(tl.float16)

    # ---- ReLU (fp16) ----
    z_h = tl.maximum(z_h, 0.0)

    # ---- GELU exact (opmath fp32, rounded to fp16) ----
    zf = z_h.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = 0.5 * zf * (1.0 + tl.math.erf(zf * INV_SQRT2))
    gel_h = gel.to(tl.float16)

    # ---- scale by scalar (opmath fp32, rounded to fp16) ----
    out = (gel_h.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS half GEMM (tensor cores)
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_act_kernel[(Mrows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, out,
            N, x.stride(0), out.stride(0),
            EPS_RMS=1e-6, EPS_LN=1e-5, SCALE=1.3529,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
