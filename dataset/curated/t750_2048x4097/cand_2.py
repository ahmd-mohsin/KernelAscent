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
    n_cols, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    out_row = out_ptr + row * stride_row

    # Pass 1: sum of squares of relu(x + b0) in fp32
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = x + b0
        v = tl.maximum(v, 0.0)
        acc += v * v
    ssum = tl.sum(acc, axis=0)
    rms = tl.math.rsqrt(ssum / n_cols + eps)

    # Pass 2: normalize, scale, add bias
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v = x + b0
        v = tl.maximum(v, 0.0)
        # match reference: (v * rms).to(fp16) * w  + b4  (fp16 arithmetic)
        normed = (v * rms).to(tl.float16)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        b4 = tl.load(b4_ptr + offs, mask=mask, other=0.0)
        y = normed * w + b4
        tl.store(out_row + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = 1024
        _fused_kernel[(n_rows,)](
            x, self.b0, self.rms3_w, self.b4, out,
            n_cols, x.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
