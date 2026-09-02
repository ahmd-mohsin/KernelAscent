import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 470
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    n_cols,
    stride_xm, stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # match reference: add performed in bf16, then softmax upcasts to fp32
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    xb = tl.where(mask, xb, float("-inf"))

    row_max = tl.max(xb, axis=0)
    num = tl.exp(xb - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4
        _bias_softmax_kernel[(n_rows,)](
            x, self.b0, out,
            n_cols,
            x.stride(0), out.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out
