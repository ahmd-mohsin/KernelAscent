import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 780
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(X, Y, G, B, W, N, stride_x, stride_y,
                  EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
                  S1: tl.constexpr, S2: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf) computed in fp32, rounded to fp16 like PyTorch
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # scale 1.0366, round to fp16
    g = (g * S1).to(tl.float16).to(tl.float32)

    # layer norm in fp32
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + EPS_LN)

    gw = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    gb = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g - mean) * rstd * gw + gb
    y = y.to(tl.float16).to(tl.float32)

    # scale 1.4943, round to fp16
    y = (y * S2).to(tl.float16).to(tl.float32)

    # RMS norm in fp32 (input already fp16 values in fp32)
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    rrms = tl.math.rsqrt(ms + EPS_RMS)
    z = (y * rrms).to(tl.float16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, y, self.ln2_g, self.ln2_b, self.rms4_w,
            N, x2.stride(0), y.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            S1=1.0366, S2=1.4943,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
