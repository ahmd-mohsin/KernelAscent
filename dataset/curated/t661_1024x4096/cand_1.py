import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 661
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_gelu2_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch on fp16 inputs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to fp16 as PyTorch would between ops
    p = p.to(tl.float16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # gelu #1 (exact, erf-based; opmath fp32 then cast to fp16)
    g1 = p * 0.5 * (1.0 + tl.math.erf(p * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)
    # gelu #2
    g2 = g1 * 0.5 * (1.0 + tl.math.erf(g1 * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, g2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores on A100)
        z = z.contiguous()
        rows, cols = z.shape
        y = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_gelu2_kernel[(rows,)](
            z, y, cols, z.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
