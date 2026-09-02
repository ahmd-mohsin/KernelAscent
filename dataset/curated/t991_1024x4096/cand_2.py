import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 991
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _ln_kernel(X, G, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _softmax_kernel(X, Y, N, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _ln_kernel[(Mrows,)](x, self.ln0_g, self.ln0_b, y, N, 1e-5,
                             BLOCK=triton.next_power_of_2(N), num_warps=8)
        z = y @ self.W1
        z = z.contiguous()
        Nout = z.shape[-1]
        out = torch.empty_like(z)
        _softmax_kernel[(Mrows,)](z, out, Nout, 1.4415,
                                  BLOCK=triton.next_power_of_2(Nout), num_warps=4)
        return out
