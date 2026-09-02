import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 141
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_relu_bias_relu_softmax(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf'))
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # relu(x)
    x = tl.maximum(x, 0.0)
    # + bias (bf16 add to match reference), then relu
    x = x + b
    x = tl.maximum(x, 0.0)

    # softmax in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 512 else 4
        _fused_relu_bias_relu_softmax[(n_rows,)](
            x, self.b1, out,
            n_cols,
            x.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
