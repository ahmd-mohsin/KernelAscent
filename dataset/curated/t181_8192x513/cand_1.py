import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 181
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_softmax_kernel(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # emulate bf16 arithmetic exactly: multiply/add with rounding to bf16 each step
    t = (x.to(tl.float32) * 1.2248).to(tl.bfloat16)
    t = (t.to(tl.float32) * 1.0339).to(tl.bfloat16)
    t = (t.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in float32 (matching PyTorch's internal fp32 accumulation)
    xf = tl.where(mask, t.to(tl.float32), float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_softmax_kernel[(n_rows,)](
            x2, self.b2, out,
            n_cols,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
