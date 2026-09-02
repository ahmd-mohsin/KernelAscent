import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 750
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, w_ptr, b4_ptr, out_ptr,
    D: tl.constexpr, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0)

    # x + b0 in fp16 (matches reference), then relu
    y = x + b0
    zero = y - y
    y = tl.where(y > zero, y, zero)

    yf = y.to(tl.float32)
    ssq = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0)
    inv = 1.0 / tl.sqrt(ssq / D + eps)

    norm = (yf * inv).to(tl.float16)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    b4 = tl.load(b4_ptr + offs, mask=mask, other=0.0)

    out = norm * w + b4  # fp16 arithmetic, matching reference
    tl.store(out_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x, self.b0, self.rms3_w, self.b4, out,
            d, x.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
