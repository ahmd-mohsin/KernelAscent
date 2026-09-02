import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 186
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_scale_gelu_softmax(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # scale (match bf16 rounding of reference: x = x * 1.3982 in bf16)
    xf = xf * 1.3982
    xf = xf.to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to bf16 like reference
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * inv_sqrt2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax over the row in fp32
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores on A100)
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_scale_gelu_softmax[(m,)](
            h, y,
            h.stride(0), y.stride(0),
            n, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
