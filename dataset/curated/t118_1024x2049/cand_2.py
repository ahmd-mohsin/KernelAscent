import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 118
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, n_cols, stride_x, stride_o,
                  BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # bias add in bf16 (matches x + b0 in bf16), then upcast for softmax math
    s = (x + b).to(tl.bfloat16)
    v = s.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))

    # softmax #1 (fp32 accumulation, output rounded to bf16 like PyTorch)
    m1 = tl.max(v, axis=0)
    e1 = tl.exp(v - m1)
    e1 = tl.where(mask, e1, 0.0)
    d1 = tl.sum(e1, axis=0)
    y1 = (e1 / d1).to(tl.bfloat16)

    # relu (no-op on non-negative values, kept for exactness)
    y1 = tl.maximum(y1, 0.0)

    # softmax #2
    v2 = y1.to(tl.float32)
    v2 = tl.where(mask, v2, float('-inf'))
    m2 = tl.max(v2, axis=0)
    e2 = tl.exp(v2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    d2 = tl.sum(e2, axis=0)
    y2 = (e2 / d2).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + cols, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.softmax(x, dim=-1)
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            return x

        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4
        _fused_kernel[(n_rows,)](
            x, self.b0, out, n_cols,
            x.stride(0), out.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out
