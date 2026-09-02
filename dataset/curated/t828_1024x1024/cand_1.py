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
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    x = tl.load(X_ptr + row * N_COLS + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)

    # relu -> scale (positive) -> relu  ==  relu(x) * SCALE
    x = tl.maximum(x, 0.0) * SCALE
    # masked lanes: set to -inf so they don't affect max/sum
    x = tl.where(mask, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Y_ptr + row * N_COLS + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


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
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_relu_scale_softmax_kernel[(n_rows,)](
            x2, y,
            N_COLS=n_cols,
            BLOCK=BLOCK,
            SCALE=1.2302,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
