import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 455
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _rms_kernel(X, W, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + cols).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    xn = (x * inv).to(tl.float16)
    w = tl.load(W + cols)
    out = xn * w
    out = out * tl.full((), 1.1255, tl.float16)
    tl.store(Y + row * stride_y + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        _rms_kernel[(m,)](
            h, self.rms1_w, y,
            h.stride(0), y.stride(0),
            N=n, BLOCK=n,
            num_warps=8,
        )
        return y
