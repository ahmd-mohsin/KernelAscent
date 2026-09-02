import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 866
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_gelu2_softmax(
    X, Y,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # first GELU (exact, erf), round to bf16 as PyTorch does between ops
    g1 = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.bfloat16).to(tl.float32)

    # second GELU
    g2 = g1 * 0.5 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation)
    g2 = tl.where(mask, g2, float('-inf'))
    m = tl.max(g2, axis=0)
    e = tl.exp(g2 - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


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
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_gelu2_softmax[(rows,)](
            x2, y, N,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
