import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 396
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _softmax_gelu2_relu_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to fp16 (intermediate storage in reference)
    p = p.to(tl.float16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf), fp32 math, fp16 storage
    g = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # gelu #2
    g2 = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g2 = g2.to(tl.float16)

    # relu
    out = tl.maximum(g2, 0.0)
    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores on A100)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_gelu2_relu_kernel[(Mrows,)](
            x, y, N, BLOCK=BLOCK, num_warps=num_warps
        )
        return y
