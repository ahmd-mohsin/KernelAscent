import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 930
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # x = x * 1.0855  (bf16 rounding to match reference)
    x = (x * 1.0855).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), computed in fp32, rounded to bf16 like PyTorch
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    x = g.to(tl.bfloat16).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # + bias (bf16 rounding)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax in fp32
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_epilogue_softmax[(m,)](
            h, self.b4, out,
            n, h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
