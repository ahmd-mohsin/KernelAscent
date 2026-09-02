import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 896
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_relu_bias_softmax(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # relu in bf16 (exact), then add in bf16 rounding to match eager semantics
    xf = tl.maximum(x.to(tl.float32), 0.0)
    v = (xf + b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)

    v = tl.where(mask, v, float('-inf'))
    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK_SIZE <= 1024 else 8

        _fused_relu_bias_softmax[(n_rows,)](
            x2, self.b1, out,
            n_cols,
            x2.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
