import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 692
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _gelu(x):
    # exact erf-based GELU (matches F.gelu default)
    return x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def _fused_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    x = _gelu(x)
    x = x * 1.2806
    x = _gelu(x)

    # softmax over the row
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = e / s

    x = _gelu(x)

    tl.store(Y + row * stride_y + cols, x.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = y * 1.2806
            y = F.gelu(y)
            y = torch.softmax(y, dim=-1)
            return F.gelu(y)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).contiguous()
        Mrows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_kernel[(Mrows,)](
            x2, y, N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.reshape(orig_shape)
