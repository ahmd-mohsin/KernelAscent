import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 185
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_relu_softmax_gelu2_kernel(
    X, Y, N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf), round back to bf16 like eager op does
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    r = tl.maximum(g, 0.0)

    # softmax over row (fp32 accumulation, output rounded to bf16)
    r = tl.where(mask, r, float('-inf'))
    m = tl.max(r, axis=0)
    e = tl.exp(r - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # gelu
    g2 = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # gelu
    g3 = 0.5 * g2 * (1.0 + tl.math.erf(g2 * INV_SQRT2))

    tl.store(Y + row * stride_y + offs, g3.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if (not x.is_cuda) or x.dtype != torch.bfloat16:
            y = F.gelu(x)
            y = torch.relu(y)
            y = torch.softmax(y, dim=-1)
            y = F.gelu(y)
            y = F.gelu(y)
            return y

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = max(triton.next_power_of_2(n), 16)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_gelu_relu_softmax_gelu2_kernel[(rows,)](
            x2, y, n,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
