import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 396
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _softmax_gelu2_relu_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # cast to fp16 (softmax output dtype) then back, to match op-by-op rounding
    p = p.to(tl.float16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # gelu 1 (exact erf, computed in fp32 as PyTorch does for half)
    g1 = p * 0.5 * (1.0 + tl.math.erf(p * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)
    # gelu 2
    g2 = g1 * 0.5 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)
    # relu
    out = tl.maximum(g2, 0.0)

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _softmax_gelu2_relu_kernel[(Mrows,)](
            x, y, N,
            x.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
