import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 472
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _softmax_scale_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    s = tl.sum(e, axis=0)
    y = e / s * SCALE
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            return torch.softmax(x, dim=-1) * 1.4851
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        Mr, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _softmax_scale_kernel[(Mr,)](
            x2, y,
            x2.stride(0), y.stride(0),
            N,
            SCALE=1.4851,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
