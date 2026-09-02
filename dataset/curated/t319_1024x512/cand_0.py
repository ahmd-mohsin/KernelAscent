import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 319
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _relu_softmax_kernel[(m,)](
            h, out, n, h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N, num_warps=8,
        )
        return out
