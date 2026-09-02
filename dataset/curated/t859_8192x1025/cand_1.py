import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 859
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_kernel(X, W0, W3, Y, D, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # --- RMSNorm #1 (float32 math, cast back to fp16, weight mul in fp16) ---
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    y = (xf * r).to(tl.float16) * w0

    # --- Softmax (fp32 internally, output fp16) ---
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    mval = tl.max(yf, axis=0)
    e = tl.exp(yf - mval)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # --- Scale by 1.2053 (opmath in fp32, round to fp16) ---
    p2 = (p.to(tl.float32) * 1.2053).to(tl.float16)

    # --- RMSNorm #2 ---
    p2f = p2.to(tl.float32)
    p2f = tl.where(mask, p2f, 0.0)
    ms2 = tl.sum(p2f * p2f, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    z = (p2f * r2).to(tl.float16) * w3

    # --- ReLU ---
    zero = tl.zeros(z.shape, dtype=tl.float16)
    z = tl.maximum(z, zero)

    tl.store(Y + row * stride_y + offs, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(n_rows,)](
            x2, self.rms0_w, self.rms3_w, y,
            d, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
