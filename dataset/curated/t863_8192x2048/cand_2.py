import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 863
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU, computed in fp32 like PyTorch's opmath
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _scale_softmax_kernel(X, Y, n_cols, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    ptr = X + row * n_cols
    x = tl.load(ptr + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # replicate: (bf16 * scalar) computed in fp32, rounded back to bf16
    x = (x * scale).to(tl.bfloat16).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * n_cols + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n = x.numel()

        # Fused GELU (elementwise, Triton)
        g = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gelu_kernel[grid](x, g, n, BLOCK=BLOCK, num_warps=4)

        # bf16 matmul via cuBLAS tensor cores (fp32 accumulate)
        y = g @ self.W1

        # Fused scale + softmax (row-wise, Triton)
        rows, cols = y.shape
        out = torch.empty_like(y)
        BLOCK_C = triton.next_power_of_2(cols)
        _scale_softmax_kernel[(rows,)](
            y, out, cols, 1.0993, BLOCK=BLOCK_C, num_warps=8
        )
        return out
