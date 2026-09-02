import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 525
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_softmax_rms_bias(X, W, B, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp16 softmax internal accumulation)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # round to fp16, as reference stores softmax output in fp16
    sm16 = sm.to(tl.float16)
    xf = sm16.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / D_
    rr = tl.math.rsqrt(ms + 1e-6)
    y16 = (xf * rr).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)

    out = y16 * w + b  # fp16 arithmetic, matching reference
    tl.store(Y + row * D_ + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_rms_bias[(m,)](
            x, self.rms1_w, self.b2, y,
            D_=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
