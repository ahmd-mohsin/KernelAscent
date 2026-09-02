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
    X, W_RMS, LN_G, LN_B, OUT,
    N, D: tl.constexpr,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then cast to fp16 (matches F.gelu on half)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # RMSNorm: stats in fp32
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)
    ms = tl.sum(gf * gf, axis=0) / D
    inv_rms = tl.math.rsqrt(ms + 1e-6)
    rn16 = (gf * inv_rms).to(tl.float16)

    # multiply by rms weight in fp16 (matches reference fp16 * fp16)
    w = tl.load(W_RMS + offs, mask=mask, other=0.0)
    y16 = rn16 * w

    # ReLU in fp16
    zero16 = tl.zeros([BLOCK], dtype=tl.float16)
    y16 = tl.maximum(y16, zero16)

    # LayerNorm: stats and affine in fp32, output fp16
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / D
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = tl.math.rsqrt(var + 1e-5)

    gamma = tl.load(LN_G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(LN_B + offs, mask=mask, other=0.0).to(tl.float32)
    out = (yf - mean) * inv_std * gamma + beta

    tl.store(OUT + row * stride_o + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(n,)](
            x2, self.rms1_w, self.ln3_g, self.ln3_b, out,
            n, d,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
