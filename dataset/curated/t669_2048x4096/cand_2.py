import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 669
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, OUT, G, B, W2, W4,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    xf = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 compute, cast to bf16) ----
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * inv_std) * g + b
    y_bf = y.to(tl.bfloat16)

    # ---- RMSNorm 1: (xf * rsqrt(mean(xf^2)+eps)).bf16() * w2 ----
    yf = y_bf.to(tl.float32)
    ms1 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + RMS_EPS)
    z_bf = (yf * r1).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    z_bf = (z_bf.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # ---- ReLU ----
    z_bf = tl.maximum(z_bf, z_bf - z_bf)  # relu in bf16 (exact)

    # ---- RMSNorm 2 ----
    zf = z_bf.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + RMS_EPS)
    o_bf = (zf * r2).to(tl.bfloat16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    o_bf = (o_bf.to(tl.float32) * w4.to(tl.float32)).to(tl.bfloat16)

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
        x = torch.matmul(x, self.W0)
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(Mrows,)](
            x, out, self.ln1_g, self.ln1_b, self.rms2_w, self.rms4_w,
            N, x.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
