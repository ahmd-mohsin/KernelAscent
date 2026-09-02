import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 256
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # relu (applied twice == once)
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N_)
        _relu_softmax_kernel[(M_,)](
            h, y, N_, h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
