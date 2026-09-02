import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 345
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _gelu_relu_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # GELU (erf-based, computed in fp32 like PyTorch opmath), then cast to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g_bf16 = g.to(tl.bfloat16)

    # ReLU in bf16
    zero = tl.zeros_like(g_bf16)
    r = tl.maximum(g_bf16, zero)

    # Softmax in fp32 (as PyTorch does for bf16 inputs)
    rf = r.to(tl.float32)
    rf = tl.where(mask, rf, float("-inf"))
    row_max = tl.max(rf, axis=0)
    e = tl.exp(rf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul, identical to reference
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _gelu_relu_softmax_kernel[(m,)](
            x, y,
            x.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
