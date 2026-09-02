import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 661
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_gelu2_kernel(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching CUDA softmax on half)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (softmax output dtype), then back to fp32 for gelu math
    p = p.to(tl.float16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # gelu #1 (exact erf), output cast to fp16 then back (matches op boundaries)
    g1 = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)

    # gelu #2
    g2 = 0.5 * g1 * (1.0 + tl.math.erf(g1 * INV_SQRT2))

    tl.store(Y + row * stride_ym + cols, g2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 4096 else 4
        _softmax_gelu2_kernel[(m,)](
            x, y,
            x.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
