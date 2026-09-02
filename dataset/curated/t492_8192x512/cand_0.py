import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 492
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_relu_gelu_softmax_kernel(
    X_ptr, Out_ptr,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # GELU (exact, erf-based) computed in fp32, then rounded to bf16 to match
    # PyTorch's bf16 gelu kernel behavior, then upcast for softmax.
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation, as PyTorch does for bf16 inputs)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores (bf16)
        h = x @ self.W0

        rows, cols = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 512 else 4
        _fused_relu_gelu_softmax_kernel[(rows,)](
            h, out,
            h.stride(0), out.stride(0),
            cols,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
