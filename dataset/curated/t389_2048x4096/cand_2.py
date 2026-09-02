import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 389
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _gelu3_softmax_relu_kernel(
    X, Y,
    n_cols,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # GELU #1 (exact erf-based, fp32 compute, round to fp16 like PyTorch)
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)
    # GELU #2
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)
    # GELU #3
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # Softmax over the row (fp32 accumulation, matches PyTorch fp16 softmax)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    # ReLU (no-op after softmax, kept for exact semantics)
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16 and x.dim() == 2:
            x = x.contiguous()
            n_rows, n_cols = x.shape
            y = torch.empty_like(x)
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _gelu3_softmax_relu_kernel[(n_rows,)](
                x, y,
                n_cols,
                x.stride(0), y.stride(0),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            return y

        # Fallback (CPU / other dtypes / other shapes)
        x = F.gelu(x)
        x = F.gelu(x)
        x = F.gelu(x)
        x = torch.softmax(x, dim=-1)
        x = torch.relu(x)
        return x
