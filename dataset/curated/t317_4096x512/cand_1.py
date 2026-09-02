import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 317
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_gelu_relu_softmax_kernel(
    X_ptr, Out_ptr,
    N,
    stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # cast to bf16 (match reference intermediate dtype), then relu
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.maximum(g, 0.0)

    # softmax over the row (fp32 accumulation like PyTorch's bf16 softmax)
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores on A100)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 16 if BLOCK_N >= 4096 else 8
        _fused_gelu_relu_softmax_kernel[(Mrows,)](
            y, out, N,
            y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
