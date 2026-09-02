import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 192
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.1857  (bf16 rounding as in reference)
    x = (x * 1.1857).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based), rounded to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # * 1.3931, rounded to bf16
    g = (g * 1.3931).to(tl.bfloat16).to(tl.float32)

    # relu (no-op on nonneg, kept for exactness)
    g = tl.maximum(g, 0.0)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul with fp32 accumulation (same as reference)
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)
        _fused_act_softmax_kernel[(Mrows,)](
            y, out,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
