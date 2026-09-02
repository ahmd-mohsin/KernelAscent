import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 835
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _softmax_relu_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    out = num / denom
    out = tl.maximum(out, 0.0)
    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            return torch.relu(torch.softmax(x, dim=-1))
        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _softmax_relu_kernel[(m,)](
            x2d, y,
            x2d.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
