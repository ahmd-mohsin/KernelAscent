import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 41
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_kernel(X, W1, W4, OUT, n_cols, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # ---- softmax 1 (float accumulation, like PyTorch half softmax) ----
    xm = tl.where(mask, xf, float('-inf'))
    m = tl.max(xm, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y16 = (e / s).to(tl.float16)

    # ---- rmsnorm 1 ----
    yf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / n_cols
    r = tl.math.rsqrt(ms + 1e-6)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    z16 = (yf * r).to(tl.float16) * w1  # fp16 multiply like PyTorch

    # ---- softmax 2 ----
    zf = z16.to(tl.float32)
    zm = tl.where(mask, zf, float('-inf'))
    m2 = tl.max(zm, axis=0)
    e2 = tl.exp(zf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p16 = (e2 / s2).to(tl.float16)

    # ---- gelu (exact, erf-based, float math) ----
    pf = p16.to(tl.float32)
    g = pf * 0.5 * (1.0 + tl.math.erf(pf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # ---- rmsnorm 2 ----
    gf = g16.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / n_cols
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    out = (gf * r2).to(tl.float16) * w4

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(n_rows,)](
            x, self.rms1_w, self.rms4_w, out,
            n_cols, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
