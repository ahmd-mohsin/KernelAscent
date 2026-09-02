import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 858
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_ln_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Out_ptr,
    stride_x, stride_o,
    N,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact erf, fp32 opmath, cast back to bf16 like PyTorch) ----
    SQRT1_2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * SQRT1_2))
    g_bf = g.to(tl.bfloat16)
    gf = g_bf.to(tl.float32)

    # ---- RMSNorm (fp32 stats, cast to bf16, then bf16*bf16 weight in fp32 opmath) ----
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + RMS_EPS)
    y = (gf * rstd).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = (y * w).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 stats) ----
    mean = tl.sum(tl.where(mask, t, 0.0), axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    r = 1.0 / tl.sqrt(var + LN_EPS)

    gamma = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = (d * r * gamma + beta).to(tl.bfloat16)
    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core bf16 GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_rms_ln_kernel[(Mrows,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, out,
            h.stride(0), out.stride(0),
            N,
            RMS_EPS=1e-6,
            LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
