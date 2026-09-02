import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 483
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _scaled_softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)

    # replicate: (x * 1.1753) in fp16, then (* 1.0586) in fp16
    t = (x.to(tl.float32) * 1.1753).to(tl.float16)
    t = (t.to(tl.float32) * 1.0586).to(tl.float16)

    v = t.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))

    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _scaled_softmax_kernel[(n_rows,)](
            x, y, n_cols,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
