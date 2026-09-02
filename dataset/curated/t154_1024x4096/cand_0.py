import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 154
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X, OUT,
    LN_G, LN_B, B3, RMS_W,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # gelu (exact, erf-based), computed in fp32 then cast to fp16 (matches PyTorch opmath)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # layer_norm in fp32 (matches PyTorch internal upcast)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = tl.math.rsqrt(var + LN_EPS)
    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * inv_std * g + b
    y = y.to(tl.float16)

    # + b3 in fp16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    y = y + b3

    # rms norm: fp32 stats, cast to fp16, then fp16 multiply by weight
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + RMS_EPS)
    z = (yf * r).to(tl.float16)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0)
    z = z * w

    tl.store(OUT + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x2, out,
            self.ln2_g, self.ln2_b, self.b3, self.rms4_w,
            N, x2.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
