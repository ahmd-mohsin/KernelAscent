import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 727
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_relu_scale_softmax_kernel(
    X_ptr, Y_ptr,
    N,                      # row length
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)

    # relu + scale
    x = tl.maximum(x, 0.0) * SCALE
    x = tl.where(mask, x, float('-inf'))

    # softmax
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.345
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        m, n = x2.shape
        y = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16

        _fused_relu_scale_softmax_kernel[(m,)](
            x2, y,
            n,
            x2.stride(0), y.stride(0),
            SCALE=1.345,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
