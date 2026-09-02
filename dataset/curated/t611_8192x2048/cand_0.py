import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 611
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    x_ptr, out_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    ptr = x_ptr + row * stride_row + cols
    x = tl.load(ptr, mask=mask, other=float('-inf')).to(tl.float32)

    # relu(relu(x)) == relu(x)
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    out = out_ptr + row * stride_row + cols
    tl.store(out, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
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

        _relu_softmax_kernel[(n_rows,)](
            x2, out,
            n_cols,
            x2.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
