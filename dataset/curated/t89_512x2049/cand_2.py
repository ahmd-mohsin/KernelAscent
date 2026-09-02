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
    stride_xm, stride_om,
    s1, s2,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # elementwise ops with fp16 rounding at each step (numerical equivalence)
    t = (x.to(tl.float16) * s1).to(tl.float16)
    t = (t + b.to(tl.float16)).to(tl.float16)
    t = (t * s2).to(tl.float16)

    # softmax in fp32 (matches torch's half softmax accumulation)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    row_max = tl.max(tf, axis=0)
    num = tl.exp(tf - row_max)
    num = tl.where(mask, num, 0.0)
    den = tl.sum(num, axis=0)
    out = (num / den).to(tl.float16)

    tl.store(out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            t = x * 1.1498
            t = t + self.b1
            t = t * 1.2338
            return torch.softmax(t, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_SIZE >= 4096 else 4
        _fused_softmax_kernel[(n_rows,)](
            x, self.b1, out,
            n_cols,
            x.stride(0), out.stride(0),
            1.1498, 1.2338,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out
