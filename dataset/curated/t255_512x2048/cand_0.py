import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 255
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_softmax_relu_scale_gelu(
    X, Y, n_cols, stride_x, stride_y,
    SCALE: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # relu (no-op on softmax output, kept for exactness)
    s = tl.maximum(s, 0.0)

    # scale
    s = s * SCALE

    # exact gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * s * (1.0 + tl.math.erf(s * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, g.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.relu(x)
            x = x * 1.4765
            return F.gelu(x)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_softmax_relu_scale_gelu[(n_rows,)](
            x2, y, n_cols, x2.stride(0), y.stride(0),
            SCALE=1.4765, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.reshape(orig_shape)
