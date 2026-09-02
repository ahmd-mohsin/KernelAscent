import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 911
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _rmsnorm_kernel(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(Y.dtype.element_ty)
    w = tl.load(W + offs, mask=mask, other=0.0)
    tl.store(Y + row * D + offs, xn * w, mask=mask)


@triton.jit
def _softmax_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = (e / s).to(Y.dtype.element_ty)
    tl.store(Y + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape
        xn = torch.empty_like(x)
        _rmsnorm_kernel[(M_,)](x, self.rms0_w, xn, D_, triton.next_power_of_2(D_), num_warps=8)
        h = xn @ self.W1
        N_ = h.shape[1]
        out = torch.empty_like(h)
        _softmax_kernel[(M_,)](h, out, N_, triton.next_power_of_2(N_), num_warps=4)
        return out
