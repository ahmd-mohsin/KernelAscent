import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 621
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    LN0_G, LN0_B, LN2_G, LN2_B, RMS_W,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    n = N.to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, round to bf16) ----
    mean0 = tl.sum(x, axis=0) / n
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / n
    rstd0 = tl.math.rsqrt(var0 + 1e-5)
    g0 = tl.load(LN0_G + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(LN0_B + cols, mask=mask, other=0.0).to(tl.float32)
    h = d0 * rstd0 * g0 + b0
    h = h.to(tl.bfloat16)

    # ---- ReLU ----
    h = tl.maximum(h, 0.0).to(tl.bfloat16)

    # ---- LayerNorm 2 (fp32 math on bf16 input, round to bf16) ----
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    mean2 = tl.sum(hf, axis=0) / n
    d2 = tl.where(mask, hf - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n
    rstd2 = tl.math.rsqrt(var2 + 1e-5)
    g2 = tl.load(LN2_G + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(LN2_B + cols, mask=mask, other=0.0).to(tl.float32)
    h2 = d2 * rstd2 * g2 + b2
    h2 = h2.to(tl.bfloat16)

    # ---- RMSNorm: xf = x.float(); xf * rsqrt(mean(xf^2)+1e-6) -> bf16; * w (bf16 mul via fp32) ----
    xf = h2.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / n
    rrms = tl.math.rsqrt(ms + 1e-6)
    r = (xf * rrms).to(tl.bfloat16)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0)
    r = (r.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # ---- GELU (exact, erf-based; fp32 opmath then round) ----
    rf = r.to(tl.float32)
    out = 0.5 * rf * (1.0 + tl.math.erf(rf * 0.7071067811865476))
    out = out.to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_norm_kernel[(rows,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.ln2_g, self.ln2_b, self.rms3_w,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
