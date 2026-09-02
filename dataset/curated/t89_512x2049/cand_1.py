import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 89
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_softmax_kernel(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    x_row_stride, out_row_stride,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # Match fp16 rounding of the reference elementwise ops
    x = (x.to(tl.float32) * S1).to(tl.float16)
    x = (x + b).to(tl.float16)
    x = (x.to(tl.float32) * S2).to(tl.float16)

    # Softmax with fp32 accumulation (matches PyTorch fp16 softmax)
    xf = tl.where(mask, x.to(tl.float32), float("-inf"))
    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(out_ptr + row * out_row_stride + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4
        _fused_softmax_kernel[(n_rows,)](
            x, self.b1, out,
            n_cols,
            x.stride(0), out.stride(0),
            S1=1.1498, S2=1.2338,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out
