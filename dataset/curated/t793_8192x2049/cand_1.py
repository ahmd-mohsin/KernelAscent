import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 793
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, RMSW_ptr, G4_ptr, B4_ptr, G5_ptr, B5_ptr, OUT_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    # ---- load matmul output (fp16 -> fp32) ----
    x = tl.load(X_ptr + base + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 math, round to fp16 as torch does for fp16 output) ----
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    x = e / denom
    x = x.to(tl.float16).to(tl.float32)

    # ---- exact GELU (erf) in fp32, cast to fp16 ----
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    xh = x.to(tl.float16)

    # ---- RMSNorm: fp32 stats, cast to fp16, then fp16 multiply by weight ----
    xf = xh.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    xn = (xf * tl.math.rsqrt(ms + 1e-6)).to(tl.float16)
    w = tl.load(RMSW_ptr + offs, mask=mask, other=0.0)
    xh = xn * w  # fp16 arithmetic (matches eager elementwise mul)

    # ---- LayerNorm 4 (fp32 math, fp16 output) ----
    xf = xh.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    y = d * tl.math.rsqrt(var + 1e-5)
    g4 = tl.load(G4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xh = (y * g4 + b4).to(tl.float16)

    # ---- LayerNorm 5 (fp32 math, fp16 output) ----
    xf = xh.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    y = d * tl.math.rsqrt(var + 1e-5)
    g5 = tl.load(G5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * g5 + b5).to(tl.float16)

    tl.store(OUT_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            h, self.rms3_w, self.ln4_g, self.ln4_b, self.ln5_g, self.ln5_b, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
