import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 3
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _gelu_scale_softmax_kernel(
    X, Y,
    n_cols,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, matching F.gelu default
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # mimic bf16 intermediate rounding of the reference (gelu output tensor is bf16)
    g = g.to(tl.bfloat16).to(tl.float32)
    g = g * SCALE
    g = g.to(tl.bfloat16).to(tl.float32)

    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    num = tl.exp(g - row_max)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            y = F.gelu(x)
            y = y * 1.0959
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _gelu_scale_softmax_kernel[(n_rows,)](
            x2, out,
            n_cols,
            x2.stride(0), out.stride(0),
            SCALE=1.0959,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
