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
    x_ptr, out_ptr,
    n_cols,
    stride_x, stride_out,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    # relu is a no-op on softmax output (all values >= 0), but keep max for safety
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            return torch.relu(torch.softmax(x, dim=-1))
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _softmax_relu_kernel[(n_rows,)](
            x2d, out, n_cols,
            x2d.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
