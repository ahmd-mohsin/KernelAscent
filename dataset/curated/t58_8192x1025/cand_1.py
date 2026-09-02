import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 58
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _relu_softmax_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    # relu (applied twice == once)
    x = tl.where(cols < N, tl.maximum(x, 0.0), float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(cols < N, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        y = torch.empty_like(h)
        Mrows, N = h.shape
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_softmax_kernel[(Mrows,)](
            h, y, h.stride(0), y.stride(0), N,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
