import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 947
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_relu_scale_softmax(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    # relu applied twice == relu once; then scale
    x = tl.maximum(x, 0.0) * SCALE
    x = tl.where(mask, x, -float('inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.1719
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        M_, N_ = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(N_)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _fused_relu_scale_softmax[(M_,)](
            x2d, y,
            x2d.stride(0), y.stride(0),
            N_,
            SCALE=1.1719,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
