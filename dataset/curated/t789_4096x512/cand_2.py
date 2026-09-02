import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 789
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_scale_relu_softmax(
    X, Y,
    n_cols,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32) * SCALE
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x * 1.3032)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK_N <= 2048 else 8
        _fused_scale_relu_softmax[(n_rows,)](
            x2, y,
            n_cols,
            x2.stride(0), y.stride(0),
            SCALE=1.3032,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
