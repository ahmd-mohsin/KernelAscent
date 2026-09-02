import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 947
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_relu_scale_softmax(
    x_ptr, out_ptr,
    n_cols,
    stride_row,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    ptr = x_ptr + row * stride_row + cols
    x = tl.load(ptr, mask=mask, other=float('-inf')).to(tl.float32)
    # relu(relu(x)) == relu(x)
    x = tl.maximum(x, 0.0)
    x = x * SCALE
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(out_ptr + row * stride_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.1719
            return torch.softmax(x, dim=-1)
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_relu_scale_softmax[(n_rows,)](
            x2, out, n_cols, x2.stride(0),
            SCALE=1.1719, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
