import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 758
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, W0, B1, W3, W4, Y,
                  D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)

    # RMSNorm 0
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    r = tl.math.rsqrt(ms + 1e-6)
    y = (xf * r).to(tl.bfloat16)
    y = y * w0

    # bias add (bf16)
    y = y + b1

    # exact GELU in fp32
    g = y.to(tl.float32)
    g = g * 0.5 * (1.0 + tl.math.erf(g * 0.7071067811865476))
    y = g.to(tl.bfloat16)

    # RMSNorm 3
    xf = y.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    r = tl.math.rsqrt(ms + 1e-6)
    y = (xf * r).to(tl.bfloat16)
    y = y * w3

    # RMSNorm 4
    xf = y.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    r = tl.math.rsqrt(ms + 1e-6)
    y = (xf * r).to(tl.bfloat16)
    y = y * w4

    tl.store(Y + row * D_ + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        _fused_kernel[(m,)](
            x, self.rms0_w, self.b1, self.rms3_w, self.rms4_w, y,
            D_=d, BLOCK=triton.next_power_of_2(d),
            num_warps=4,
        )
        return y
