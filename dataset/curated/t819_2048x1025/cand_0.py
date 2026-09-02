import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 819
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _fused_relu_scale_softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    x_row_stride, y_row_stride,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    # relu (idempotent, one application suffices) in fp32 (matches bf16 relu exactly)
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)
    # scalar multiply: PyTorch computes in fp32 (opmath) then rounds to bf16
    y = xf * scale
    y = y.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    y = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y, axis=0)
    num = tl.exp(y - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y_ptr + row * y_row_stride + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.2572
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 8192:
            num_warps = 16

        _fused_relu_scale_softmax_kernel[(n_rows,)](
            x2, out,
            n_cols,
            x2.stride(0), out.stride(0),
            1.2572,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
