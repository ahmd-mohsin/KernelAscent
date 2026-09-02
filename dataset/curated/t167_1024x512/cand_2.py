import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 167
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_gelu2_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf), round to bf16 like the reference op boundary
    g1 = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g1 = g1.to(tl.bfloat16).to(tl.float32)

    # gelu #2, round to bf16
    g2 = 0.5 * g1 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 accumulation
    g2m = tl.where(mask, g2, float('-inf'))
    row_max = tl.max(g2m, axis=0)
    e = tl.exp(g2m - row_max)
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
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        Mrows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _fused_gelu2_softmax_kernel[(Mrows,)](
            x2, y,
            x2.stride(0), y.stride(0),
            N,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
