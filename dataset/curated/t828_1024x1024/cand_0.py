import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 828
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_relu_scale_softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # relu -> scale (positive) -> relu  ==  relu(x) * scale
    x = tl.maximum(x, 0.0) * SCALE
    # masked lanes were -inf -> relu makes 0 * scale = 0; but we want them excluded
    x = tl.where(mask, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.2302
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
        n_rows, n_cols = x2d.shape
        y = torch.empty_like(x2d)

        BLOCK_N = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16

        _fused_relu_scale_softmax_kernel[(n_rows,)](
            x2d, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            SCALE=1.2302,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
