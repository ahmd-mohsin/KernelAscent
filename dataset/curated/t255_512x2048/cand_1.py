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
    IS_FP16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's fp16 softmax)
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    den = tl.sum(num, axis=0)
    y = num / den

    if IS_FP16:
        # replicate intermediate fp16 rounding of the reference implementation
        y = y.to(tl.float16).to(tl.float32)

    # relu (softmax output is non-negative; kept for exact semantics)
    y = tl.maximum(y, 0.0)

    # scale
    y = y * 1.4765
    if IS_FP16:
        y = y.to(tl.float16).to(tl.float32)

    # exact GELU (erf-based, matching F.gelu default)
    g = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))

    if IS_FP16:
        g = g.to(tl.float16)
    tl.store(Y + row * stride_y + offs, g, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = torch.relu(y)
            y = y * 1.4765
            return F.gelu(y)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        n_rows, n_cols = x2.shape

        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_softmax_relu_scale_gelu[(n_rows,)](
            x2, out, n_cols,
            x2.stride(0), out.stride(0),
            IS_FP16=(x.dtype == torch.float16),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)
