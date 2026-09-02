import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 767
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * N + offs

    x = tl.load(ptr, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # rmsnorm (fp32)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * r).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    xh = xh * w

    # gelu (erf, exact)
    g = xh.to(tl.float32)
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # softmax 2
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    m2 = tl.max(gf, axis=0)
    e2 = tl.exp(gf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)
    out = out * 1.2619

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _fused_kernel[(Mrows,)](x, self.rms2_w, y, N=N, BLOCK=1024, num_warps=8)
        return y
