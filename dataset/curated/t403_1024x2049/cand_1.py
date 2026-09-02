import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 403
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, g_ptr, b_ptr, out_ptr,
    N, stride_row,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row * stride_row

    # ---- load input, compute exact GELU in fp32 (matches PyTorch opmath) ----
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    # cast back to fp16 (as PyTorch's gelu output), then re-promote for RMS
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)

    # ---- RMSNorm ----
    ms = tl.sum(gf * gf, axis=0) / N
    r = tl.math.rsqrt(ms + RMS_EPS)
    y16 = (gf * r).to(tl.float16)

    # ---- multiply by rms weight (half mul with float opmath) ----
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z16 = (y16.to(tl.float32) * w).to(tl.float16)

    # ---- ReLU ----
    z16 = tl.maximum(z16, tl.zeros_like(z16))
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)

    # ---- LayerNorm (stats in fp32) ----
    mean = tl.sum(zf, axis=0) / N
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.math.rsqrt(var + LN_EPS)

    gamma = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * inv * gamma + beta).to(tl.float16)

    tl.store(out_ptr + base + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.rms1_w, self.ln3_g, self.ln3_b, out,
            N, x2.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
