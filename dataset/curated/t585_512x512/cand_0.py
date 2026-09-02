import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 585
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / D_
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * rstd).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # scale, cast back to fp16 to match fp16 op semantics
    y = (y.to(tl.float32) * 1.4123).to(tl.float16)

    # relu
    y = tl.maximum(y, 0.0)

    # scale
    y = (y.to(tl.float32) * 1.0481).to(tl.float16)

    # exact gelu (computed in fp32 internally, like PyTorch half gelu)
    yf = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    out = g.to(tl.float16)

    tl.store(Y + row * D_ + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        _fused_kernel[(m,)](x, self.rms0_w, y, D_=d, BLOCK=triton.next_power_of_2(d),
                            num_warps=4)
        return y
