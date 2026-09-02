import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 771
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _gelu_bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # GELU (exact, erf-based) computed in fp32 like PyTorch's opmath for bf16
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    # round to bf16 (as gelu output would be), then add bias in bf16
    g_bf = g.to(tl.bfloat16)
    y = g_bf + b.to(tl.bfloat16)

    # softmax in fp32 accumulation
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (bf16, tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _gelu_bias_softmax_kernel[(m,)](
            h, self.b2, out,
            h.stride(0), out.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
