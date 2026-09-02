import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 314
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_act_softmax_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (fp32 math, rounded to fp16 like PyTorch half elementwise ops)
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # relu (exact on fp16)
    g = tl.maximum(g, 0.0)

    # gelu again
    g2 = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)

    # scalar multiply (fp32 math, rounded to fp16)
    g2 = g2 * 1.2863
    g2 = g2.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation (as PyTorch does for fp16 input)
    g2 = tl.where(mask, g2, float('-inf'))
    m = tl.max(g2, 0)
    e = tl.exp(g2 - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = e / s

    tl.store(Y + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_act_softmax_kernel[(Mrows,)](h, y, N, BLOCK=BLOCK, num_warps=num_warps)
        return y
