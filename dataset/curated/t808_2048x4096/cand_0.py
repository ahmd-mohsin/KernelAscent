import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 808
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, D_: tl.constexpr, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu (fp16)
    x = tl.maximum(x, 0.0)

    # rmsnorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w  # fp16 mul (matches torch)

    # relu (fp16)
    y = tl.maximum(y, 0.0)

    # scale in fp16 (matches torch half scalar mul)
    scale = tl.full((1,), 1.3505, tl.float16)
    y = y * scale

    # softmax with fp32 accumulation (matches torch half softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D_)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(M_,)](
            x, self.rms1_w, y,
            D_, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
