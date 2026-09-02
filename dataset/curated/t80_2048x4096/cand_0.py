import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 80
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _softmax_gelu_x2_kernel(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, output rounded to bf16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16).to(tl.float32)

    # gelu 1 (exact erf), rounded to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = y * 0.5 * (1.0 + tl.math.erf(y * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax 2
    g = tl.where(mask, g, -float('inf'))
    m2 = tl.max(g, axis=0)
    e2 = tl.exp(g - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = (e2 / s2).to(tl.bfloat16).to(tl.float32)

    # gelu 2
    g2 = y2 * 0.5 * (1.0 + tl.math.erf(y2 * INV_SQRT2))

    tl.store(Y + row * stride_ym + cols, g2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul, identical to reference
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_gelu_x2_kernel[(Mrows,)](
            h, out,
            h.stride(0), out.stride(0),
            N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
