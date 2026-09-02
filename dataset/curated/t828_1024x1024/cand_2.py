import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 828
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_relu_scale_softmax(
    x_ptr, out_ptr,
    n_cols,
    stride_row,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    # relu -> * scale (in fp32, matching PyTorch's opmath for half), round to fp16
    xf = x.to(tl.float32)
    y = tl.maximum(xf, 0.0) * SCALE
    y = y.to(tl.float16)  # single rounding, matches half tensor result
    # second relu is a no-op on non-negative values, but keep for exactness
    y = tl.maximum(y, 0.0)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    num = tl.exp(yf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    result = num / denom

    tl.store(out_ptr + row * stride_row + cols, result.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = torch.relu(x)
            x = x * 1.2302
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_relu_scale_softmax[(n_rows,)](
            x2d, out,
            n_cols,
            x2d.stride(0),
            SCALE=1.2302,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
