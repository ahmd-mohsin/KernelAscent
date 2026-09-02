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
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)

    # x = x + b0 (fp16 rounding), then x = x + b1 (fp16 rounding)
    x = (x + b0).to(tl.float16)
    x = (x + b1).to(tl.float16)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = tl.math.rsqrt(ms + 1e-6)

    # cast normalized value back to fp16, multiply by weight in fp16
    xn = (xf * rrms).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (xn * w).to(tl.float16)

    # scalar multiply: opmath float, result rounded to fp16
    yf = y.to(tl.float32) * 1.287
    y2 = yf.to(tl.float16).to(tl.float32)

    # exact GELU in fp32 (matches PyTorch half gelu with float opmath)
    g = 0.5 * y2 * (1.0 + tl.math.erf(y2 * 0.7071067811865476))

    tl.store(Y + row * stride_y + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(m,)](
            x2, self.b0, self.b1, self.rms2_w, y,
            x2.stride(0), y.stride(0),
            N=n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
