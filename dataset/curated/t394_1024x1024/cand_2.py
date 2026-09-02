import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 394
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, OUT_ptr,
    RMS1_ptr, LN2G_ptr, LN2B_ptr, RMS4_ptr, RMS5_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)
    Nf = N.to(tl.float32)

    # ---- RMSNorm 1 ----
    ms = tl.sum(xf * xf, axis=0) / Nf
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * rstd).to(tl.float16)
    w1 = tl.load(RMS1_ptr + cols, mask=mask, other=0.0)
    x16 = y16 * w1  # fp16 multiply, matches reference

    # ---- LayerNorm ----
    xf = x16.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / Nf
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / Nf
    rstd_ln = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN2G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN2B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    xln = (xf - mean) * rstd_ln * g + b
    x16 = xln.to(tl.float16)

    # ---- GELU (exact, erf) ----
    xf = x16.to(tl.float32)
    xg = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x16 = xg.to(tl.float16)

    # ---- RMSNorm 4 ----
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / Nf
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * rstd).to(tl.float16)
    w4 = tl.load(RMS4_ptr + cols, mask=mask, other=0.0)
    x16 = y16 * w4

    # ---- RMSNorm 5 ----
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / Nf
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * rstd).to(tl.float16)
    w5 = tl.load(RMS5_ptr + cols, mask=mask, other=0.0)
    x16 = y16 * w5

    tl.store(OUT_ptr + row * stride_o + cols, x16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_post_kernel[(Mrows,)](
            x, out,
            self.rms1_w, self.ln2_g, self.ln2_b, self.rms4_w, self.rms5_w,
            N, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
