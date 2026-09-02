import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 43
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_post_kernel(Y, OUT, N, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    ptr = Y + row * stride_row + cols

    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf) computed in fp32, rounded to fp16 (matches PyTorch half opmath)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax 1 (fp32 accumulation, fp16 output rounding)
    g_masked = tl.where(mask, g, float('-inf'))
    m1 = tl.max(g_masked, axis=0)
    e1 = tl.exp(g_masked - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = e1 / tl.sum(e1, axis=0)
    s1 = s1.to(tl.float16).to(tl.float32)

    # ReLU (softmax output is non-negative -> identity, kept for exactness)
    s1 = tl.maximum(s1, 0.0)

    # Softmax 2
    s1_masked = tl.where(mask, s1, float('-inf'))
    m2 = tl.max(s1_masked, axis=0)
    e2 = tl.exp(s1_masked - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = e2 / tl.sum(e2, axis=0)
    s2 = s2.to(tl.float16).to(tl.float32)

    # Final GELU
    o = s2 * 0.5 * (1.0 + tl.math.erf(s2 * INV_SQRT2))

    tl.store(OUT + row * stride_row + cols, o.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM
        y = y.contiguous()
        rows, cols = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(cols)
        _fused_post_kernel[(rows,)](
            y, out, cols, y.stride(0), BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )
        return out
