import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 669
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_ln_rms_relu_rms(
    X, OUT, G, B, W2, W4,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, bf16 output like F.layer_norm on bf16) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd) * g + b
    y_bf = y.to(tl.bfloat16)

    # ---- RMSNorm 2: fp32 norm, cast to bf16, multiply by weight in bf16 ----
    yf = y_bf.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + RMS_EPS)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    z_bf = (yf * r2).to(tl.bfloat16) * w2

    # ---- ReLU (bf16) ----
    z_bf = tl.maximum(z_bf, z_bf * 0)

    # ---- RMSNorm 4 ----
    zf = z_bf.to(tl.float32)
    ms4 = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0) / N
    r4 = 1.0 / tl.sqrt(ms4 + RMS_EPS)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    o_bf = (zf * r4).to(tl.bfloat16) * w4

    tl.store(OUT + row * stride_o + cols, o_bf, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 with fp32 accumulate
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_rms_relu_rms[(Mrows,)](
            h, out, self.ln1_g, self.ln1_b, self.rms2_w, self.rms4_w,
            N, h.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
