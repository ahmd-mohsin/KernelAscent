import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 179
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, OUT,
    RMS_W, G3, B3, G4, B4,
    N, stride_x, stride_o,
    EPS_RMS: tl.constexpr, EPS_LN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm (fp32 math, then cast to fp16 like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    y16 = (xf * r).to(tl.float16)

    # * rms1_w (half*half with float opmath, rounded to half)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0).to(tl.float32)
    z16 = (y16.to(tl.float32) * w).to(tl.float16)

    # * 1.4245 (float opmath, rounded to half)
    s16 = (z16.to(tl.float32) * SCALE).to(tl.float16)

    # LayerNorm 3 (fp32 internally, output fp16)
    a = s16.to(tl.float32)
    mu = tl.sum(tl.where(mask, a, 0.0), axis=0) / N
    d = tl.where(mask, a - mu, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    a16 = (d * inv * g3 + b3).to(tl.float16)

    # LayerNorm 4
    b = a16.to(tl.float32)
    mu2 = tl.sum(tl.where(mask, b, 0.0), axis=0) / N
    d2 = tl.where(mask, b - mu2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(var2 + EPS_LN)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out16 = (d2 * inv2 * g4 + b4).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_kernel[(Mrows,)](
            x, out,
            self.rms1_w, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b,
            N, x.stride(0), out.stride(0),
            EPS_RMS=1e-6, EPS_LN=1e-5, SCALE=1.4245,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
