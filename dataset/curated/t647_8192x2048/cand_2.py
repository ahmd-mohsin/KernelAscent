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
    X, RMSW, LNG, LNB, OUT,
    N, stride_x, stride_o,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (compute in fp32, round to bf16 like reference) ----
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + RMS_EPS)
    y = (x * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(RMSW + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y * w).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, single rounding to bf16) ----
    mean = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    zc = tl.where(mask, z - mean, 0.0)
    var = tl.sum(zc * zc, axis=0) / N
    inv = tl.math.rsqrt(var + LN_EPS)
    g = tl.load(LNG + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LNB + cols, mask=mask, other=0.0).to(tl.float32)
    h = (zc * inv * g + b).to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 internally, round to bf16) ----
    h_masked = tl.where(mask, h, float('-inf'))
    mx = tl.max(h_masked, axis=0)
    e = tl.exp(h - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (erf, fp32 internally) ----
    out = p * 0.5 * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_softmax_gelu[(Mrows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, out,
            N, x.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
