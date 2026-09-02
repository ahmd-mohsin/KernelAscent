import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 939
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(Out_ptr + row * stride_om + cols, y.to(Out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores on bf16)
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _bias_softmax_kernel[(m,)](
            y, self.b1, out,
            y.stride(0), out.stride(0),
            n, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
