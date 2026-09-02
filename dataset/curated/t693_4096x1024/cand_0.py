import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 693
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # emulate bf16 multiply (reference multiplies in bf16 before softmax)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # bias add (cheap elementwise), matmul via cuBLAS tensor cores
        h = (x + self.b0) @ self.W1
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _scale_softmax_kernel[(Mrows,)](
            h, out,
            N, h.stride(0), out.stride(0),
            1.2082,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
