import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 691
M, D, DT = 1024, 1025, torch.bfloat16


@triton.jit
def _fused_relu_bias_softmax_kernel(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    x_row_stride, out_row_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # relu(x) + b, then relu (relu twice == relu once)
    v = tl.maximum(x, 0.0) + b
    v = tl.maximum(v, 0.0)

    v = tl.where(mask, v, float('-inf'))
    vmax = tl.max(v, axis=0)
    e = tl.exp(v - vmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * out_row_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        orig_shape = x.shape
        x2 = x.view(-1, n_cols)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_bias_softmax_kernel[(rows,)](
            x2, self.b1, out,
            n_cols,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
