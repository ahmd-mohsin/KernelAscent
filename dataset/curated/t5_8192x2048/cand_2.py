import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 5
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, B0, B1, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)

    # x = x + b0 (fp16 rounding), then x = x + b1 (fp16 rounding)
    t = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.float16)
    t = (t.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)

    # RMSNorm in fp32
    xf = t.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    norm = (xf * inv).to(tl.float16)

    # * rms2_w (fp16 rounding)
    y = (norm.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    # * 1.287 (fp16 rounding)
    y = (y.to(tl.float32) * 1.287).to(tl.float16)

    # exact GELU with fp32 opmath, round to fp16
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    g = g.to(tl.float16)

    tl.store(Y + row * stride_y + cols, g, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x, self.b0, self.b1, self.rms2_w, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
