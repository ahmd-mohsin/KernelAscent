import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 628
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_softmax_gelu3_softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    stride_xm, stride_ym,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)

    # ---- softmax #1 (fp32 math, round to fp16 like PyTorch half softmax) ----
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    s = tl.sum(e, axis=0)
    y = e / s
    y = y.to(tl.float16).to(tl.float32)

    SQRT1_2: tl.constexpr = 0.7071067811865476

    # ---- gelu x3 (fp32 opmath, round to fp16 after each, like PyTorch) ----
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQRT1_2))
    y = y.to(tl.float16).to(tl.float32)
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQRT1_2))
    y = y.to(tl.float16).to(tl.float32)
    y = 0.5 * y * (1.0 + tl.math.erf(y * SQRT1_2))
    y = y.to(tl.float16).to(tl.float32)

    # ---- softmax #2 ----
    y = tl.where(mask, y, float('-inf'))
    row_max2 = tl.max(y, axis=0)
    e2 = tl.exp(y - row_max2)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(Y_ptr + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if (not x.is_cuda) or x.dtype != torch.float16 or x.dim() != 2:
            y = torch.softmax(x, dim=-1)
            y = F.gelu(y)
            y = F.gelu(y)
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = max(triton.next_power_of_2(n_cols), 16)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 8192:
            num_warps = 16
        _fused_softmax_gelu3_softmax_kernel[(n_rows,)](
            x, out,
            n_cols,
            x.stride(0), out.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out
