import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 688
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Out_ptr,
    N, stride_row,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    ptrs = X_ptr + row * stride_row + cols
    x = tl.load(ptrs, mask=mask, other=float('-inf')).to(tl.float32)

    # scale
    x = x * SCALE
    # relu
    x = tl.maximum(x, 0.0)
    # exact gelu: x * 0.5 * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    x = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    # mask out-of-bounds for softmax
    x = tl.where(mask, x, float('-inf'))

    # softmax
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    out_ptrs = Out_ptr + row * stride_row + cols
    tl.store(out_ptrs, y.to(Out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already optimal on A100 tensor cores)
        y = x @ self.W0
        y = y.contiguous()
        rows, cols = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 512 else 4
        _fused_act_softmax_kernel[(rows,)](
            y, out,
            cols, y.stride(0),
            SCALE=1.0642,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
