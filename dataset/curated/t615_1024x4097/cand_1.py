import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 615
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_gelu_ln_rms_kernel(
    X, G, B, W, OUT,
    D, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    xp = X + row * stride_x
    op = OUT + row * stride_o

    INV_SQRT2 = 0.7071067811865476

    # ---- Pass 1: gelu (rounded to fp16 like eager elementwise op), stats for layer_norm ----
    s = 0.0
    ss = 0.0
    for start in range(0, D, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < D
        x = tl.load(xp + offs, mask=m, other=0.0).to(tl.float32)
        g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
        g = g.to(tl.float16).to(tl.float32)  # match fp16 rounding of eager gelu output
        g = tl.where(m, g, 0.0)
        s += tl.sum(g, axis=0)
        ss += tl.sum(g * g, axis=0)

    mean = s / D
    var = ss / D - mean * mean
    rstd = tl.math.rsqrt(var + 1e-5)

    # ---- Pass 2: layer_norm output (rounded to fp16), accumulate sum of squares for RMS ----
    ssy = 0.0
    for start in range(0, D, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < D
        x = tl.load(xp + offs, mask=m, other=0.0).to(tl.float32)
        g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
        g = g.to(tl.float16).to(tl.float32)
        ga = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        be = tl.load(B + offs, mask=m, other=0.0).to(tl.float32)
        y = (g - mean) * rstd * ga + be
        y16f = y.to(tl.float16).to(tl.float32)  # layer_norm output stored as fp16 in eager
        y16f = tl.where(m, y16f, 0.0)
        ssy += tl.sum(y16f * y16f, axis=0)

    inv_rms = tl.math.rsqrt(ssy / D + 1e-6)

    # ---- Pass 3: apply RMS normalization and rms2 weight (fp16 multiply, as in eager) ----
    for start in range(0, D, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < D
        x = tl.load(xp + offs, mask=m, other=0.0).to(tl.float32)
        g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
        g = g.to(tl.float16).to(tl.float32)
        ga = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        be = tl.load(B + offs, mask=m, other=0.0).to(tl.float32)
        y = (g - mean) * rstd * ga + be
        y16f = y.to(tl.float16).to(tl.float32)
        normed16 = (y16f * inv_rms).to(tl.float16)
        w = tl.load(W + offs, mask=m, other=0.0)  # fp16
        out = normed16 * w  # fp16 multiply, matching eager
        tl.store(op + offs, out, mask=m)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = 1024
        _fused_gelu_ln_rms_kernel[(m,)](
            x2d, self.ln1_g, self.ln1_b, self.rms2_w, out,
            d, x2d.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
