import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 506
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, LN_G, LN_B, RMS_W, B3, OUT,
    N, stride_x, stride_o,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, bf16 rounding at output) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.rsqrt(var + EPS_LN)

    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32 math per reference, cast to bf16, then *w in bf16 opmath) ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    r = tl.rsqrt(ms + EPS_RMS)
    z = (y * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z * w).to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU (erf), fp32 opmath, round to bf16 ----
    SQRT1_2: tl.constexpr = 0.7071067811865476
    gel = z * 0.5 * (1.0 + tl.erf(z * SQRT1_2))
    gel = gel.to(tl.bfloat16).to(tl.float32)

    # ---- bias add (fp32 opmath, round to bf16) ----
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (gel + b3).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, self.b3, out,
            N, x2.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
