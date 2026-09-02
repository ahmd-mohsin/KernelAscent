import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 270
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _scaled_softmax_kernel(
    X, Y,
    n_cols,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # match reference: multiply happens in bf16, then softmax accumulates in fp32
    xs = (x.to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)

    row_max = tl.max(xs, axis=0)
    e = tl.exp(xs - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _scaled_softmax_kernel[(m,)](
            x2, y, n,
            x2.stride(0), y.stride(0),
            SCALE=1.0211,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
