import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 915
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _double_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # first softmax (fp32 accumulate, like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to bf16 to match intermediate storage in reference
    p = p.to(tl.bfloat16).to(tl.float32)
    p = tl.where(mask, p, -float('inf'))

    # second softmax
    m2 = tl.max(p, axis=0)
    e2 = tl.exp(p - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2:
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _double_softmax_kernel[(Mrows,)](
            x, y, N, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps
        )
        return y
