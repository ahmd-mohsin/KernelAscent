import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 876
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _softmax_gelu_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (numerically stable)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    ex = tl.exp(x)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    p = ex / denom

    # exact GELU: 0.5 * p * (1 + erf(p / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))

    tl.store(Y + row * stride_ym + cols, g.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0  # (M, 512), fp16 tensor-core matmul
        z = z.contiguous()
        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _softmax_gelu_kernel[(m,)](
            z, out,
            z.stride(0), out.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
