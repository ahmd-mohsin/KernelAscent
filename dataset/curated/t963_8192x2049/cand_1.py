import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 963
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _fused_kernel(
    X, W, G, B, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    SCALE: tl.constexpr,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # x = x * 1.3694 in fp16 (match reference dtype behavior), then upcast
    x = (x * SCALE).to(tl.float16)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = tl.math.rsqrt(ms + RMS_EPS)
    xn = (xf * inv).to(tl.float16)

    # multiply by rms weight in fp16
    w = tl.load(W + cols, mask=mask, other=0.0)
    v16 = (xn * w).to(tl.float16)
    v = v16.to(tl.float32)

    # LayerNorm stats in fp32
    mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / D_
    diff = tl.where(mask, v - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    rstd = tl.math.rsqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((v - mean) * rstd) * g + b
    y16 = y.to(tl.float16)

    # GELU (erf, computed in fp32 as aten does for half)
    yf = y16.to(tl.float32)
    out = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms1_w, self.ln2_g, self.ln2_b, y,
            x2.stride(0), y.stride(0),
            d, 1.3694, 1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
