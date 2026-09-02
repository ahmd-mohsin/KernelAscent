import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 66
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, OUT, W1, W3, G4, B4,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # ---- RMSNorm 1 ----
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    y = (xf * inv).to(tl.float16)  # cast to fp16 like reference
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)  # fp16
    y = (y * w1)  # fp16 multiply

    # ---- GELU (exact, erf) computed in fp32, rounded back to fp16 ----
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    y = g.to(tl.float16)

    # ---- RMSNorm 3 ----
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms2 = tl.sum(yf * yf, axis=0) / N
    inv2 = tl.math.rsqrt(ms2 + 1e-6)
    y2 = (yf * inv2).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    y2 = (y2 * w3)  # fp16 multiply

    # ---- LayerNorm 4 (fp32 internal, like PyTorch mixed-dtype LN) ----
    zf = y2.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean = tl.sum(zf, axis=0) / N
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd) * g4 + b4
    out16 = out.to(tl.float16)

    # ---- final scale (fp32 opmath, cast fp16 - matches PyTorch half scalar mul)
    res = (out16.to(tl.float32) * 1.2037).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul on tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_norm_kernel[(m,)](
            h, out,
            self.rms1_w, self.rms3_w, self.ln4_g, self.ln4_b,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
