import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 987
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _gelu2_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # First GELU (exact erf), round to bf16 to match elementwise op dtype
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.bfloat16).to(tl.float32)

    # Second GELU
    g2 = 0.5 * g1 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    g2 = tl.where(mask, g2, float('-inf'))
    row_max = tl.max(g2, axis=0)
    num = tl.exp(g2 - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

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

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _gelu2_softmax_kernel[(m,)](
            x, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
