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
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # First softmax (fp32 accumulation, as torch does for bf16 input)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1

    # Round to bf16 to match reference intermediate storage
    p1 = p1.to(tl.bfloat16).to(tl.float32)
    p1 = tl.where(mask, p1, float('-inf'))

    # Second softmax
    m2 = tl.max(p1, axis=0)
    e2 = tl.exp(p1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    tl.store(Y + row * stride_ym + cols, p2.to(tl.bfloat16), mask=mask)


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
        m, n = x2d.shape
        y = torch.empty_like(x2d)

        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16

        _double_softmax_kernel[(m,)](
            x2d, y,
            x2d.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
