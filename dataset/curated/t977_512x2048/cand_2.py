import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 977
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # relu(x)
    x = tl.maximum(x, 0.0)
    # x + b
    x = x + b
    # relu
    x = tl.maximum(x, 0.0)

    # softmax
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_y + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x, self.b1, y, n,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
