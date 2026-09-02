import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 945
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _softmax_bias_gelu_kernel(
    X, B, OUT,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch half softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # cast to fp16 (softmax output dtype), add bias in fp16 arithmetic
    sm_h = sm.to(tl.float16)
    b = tl.load(B + cols, mask=mask, other=0.0)
    z_h = sm_h + b  # fp16 add, same as reference

    # gelu computed in fp32 (opmath), output fp16 — matches PyTorch half gelu
    z = z_h.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(OUT + row * stride_om + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_bias_gelu_kernel[(m,)](
            y, self.b2, out,
            n, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
