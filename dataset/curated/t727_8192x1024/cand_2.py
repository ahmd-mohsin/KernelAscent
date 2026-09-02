import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 727
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_relu_scale_softmax(
    x_ptr, out_ptr,
    n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    ptr = x_ptr + row * stride_row + cols
    x = tl.load(ptr, mask=mask, other=float('-inf'))

    # relu (in input dtype semantics)
    x = tl.maximum(x, 0.0)

    # scale in fp32, round back to bf16 to match eager intermediate
    xf = x.to(tl.float32) * 1.345
    x_bf = xf.to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch internal accumulation)
    v = tl.where(mask, x_bf.to(tl.float32), float('-inf'))
    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
        _fused_relu_scale_softmax[(n_rows,)](
            x, out, n_cols, x.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
