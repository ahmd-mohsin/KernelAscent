import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 232
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_relu_scale_softmax(
    X, Y,
    n_cols,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu in input dtype
    x = tl.maximum(x, 0.0)
    # scalar mul: compute in fp32, round back to fp16 (matches PyTorch opmath)
    xf = x.to(tl.float32) * SCALE
    xh = xf.to(tl.float16)
    # softmax in fp32 (matches PyTorch half softmax accumulation)
    v = xh.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))
    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_relu_scale_softmax[(m,)](
            x, y, n,
            x.stride(0), y.stride(0),
            SCALE=1.2092,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
