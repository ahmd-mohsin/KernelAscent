import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 324
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_bias_relu_softmax(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    v = x + b
    v = tl.maximum(v, 0.0)
    v = tl.where(mask, v, float('-inf'))

    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, cols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, cols)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_bias_relu_softmax[(x2.shape[0],)](
            x2, self.b0, out,
            cols, x2.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view_as(x)
