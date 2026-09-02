import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 994
M, D, DT = 1024, 4096, torch.bfloat16


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
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    denom = tl.sum(e, axis=0)
    y = e / denom
    # relu is a no-op on softmax output (nonnegative), but keep max for exactness
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            return torch.relu(torch.softmax(x, dim=-1))
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
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
        return y.reshape(orig_shape)
