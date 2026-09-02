import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 835
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _softmax_relu_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    # softmax output is already non-negative; relu is identity
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        Mrows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _softmax_relu_kernel[(Mrows,)](
            x2, y,
            x2.stride(0), y.stride(0),
            N,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
