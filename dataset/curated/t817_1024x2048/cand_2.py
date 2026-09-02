import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 817
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_relu_scale_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # relu + scale in fp32 (matches CUDA opmath for half), round back to fp16
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)
    xf = xf * SCALE
    xh = xf.to(tl.float16)  # rounding step to match eager intermediate
    xf = xh.to(tl.float32)
    xf = tl.maximum(xf, 0.0)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
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
        if not x.is_cuda or x.dtype != torch.float16:
            x = torch.relu(x)
            x = x * 1.0204
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 2048 else 8
        _fused_relu_scale_softmax_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n,
            SCALE=1.0204,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
