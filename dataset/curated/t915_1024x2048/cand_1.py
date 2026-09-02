import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 915
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _double_softmax_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 math, then round to bf16 to match PyTorch's intermediate)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = e1 / s1
    y1 = y1.to(tl.bfloat16).to(tl.float32)
    y1 = tl.where(mask, y1, float('-inf'))

    # second softmax
    m2 = tl.max(y1, axis=0)
    e2 = tl.exp(y1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = e2 / s2

    tl.store(Y + row * stride_ym + cols, y2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        Mrows, N = x2d.shape
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _double_softmax_kernel[(Mrows,)](
            x2d, y, N,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
