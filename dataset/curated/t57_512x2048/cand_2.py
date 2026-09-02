import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 57
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(x_ptr, out_ptr, n_cols, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    ptr = x_ptr + row * stride_row + cols

    x = tl.load(ptr, mask=mask, other=0.0)

    # relu (in fp16, same as torch)
    x = tl.where(x > 0, x, x * 0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1: compute in fp32, round back to fp16 (matches torch opmath)
    xf = x.to(tl.float32)
    xf = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # gelu #2
    xf = x.to(tl.float32)
    xf = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # softmax in fp32 accumulation
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](x, out, d, x.stride(0), BLOCK=BLOCK, num_warps=num_warps)
        return out
