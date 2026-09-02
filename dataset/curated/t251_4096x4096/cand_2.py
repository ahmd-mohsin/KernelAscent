import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 251
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_gelu2_relu_softmax(
    X_ptr, Y_ptr,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # GELU #1 (exact erf variant), computed in fp32 then rounded back to bf16
    # to match PyTorch's opmath behavior on bf16 tensors.
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # GELU #2
    g = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ReLU (exact on bf16, no rounding needed)
    g = tl.maximum(g, 0.0)

    # Softmax over the row in fp32 (matches PyTorch's accscalar behavior)
    g = tl.where(mask, g, float("-inf"))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_row + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores on A100 for bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        m, n = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _fused_gelu2_relu_softmax[(m,)](
            h, out,
            h.stride(0),
            N=n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
