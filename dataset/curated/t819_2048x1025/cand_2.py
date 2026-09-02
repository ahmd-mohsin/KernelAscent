import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 819
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _fused_relu_scale_softmax(
    X, Y,
    N,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # relu(relu(x)) == relu(x); scale computed in fp32 then rounded to bf16
    t = tl.maximum(xf, 0.0) * SCALE
    t = t.to(tl.bfloat16).to(tl.float32)
    t = tl.where(mask, t, float('-inf'))

    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.2572
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _fused_relu_scale_softmax[(m,)](
            x2, y, n,
            x2.stride(0), y.stride(0),
            SCALE=1.2572,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
