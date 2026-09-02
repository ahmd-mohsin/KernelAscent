import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 491
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _relu_softmax_kernel(x_ptr, y_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(y_ptr + row * stride + cols, out.to(y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n = x.numel()
        xg = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gelu_kernel[grid](x, xg, n, BLOCK=BLOCK, num_warps=4)

        y = xg @ self.W1  # cuBLAS bf16 matmul (tensor cores)

        rows, cols = y.shape
        out = torch.empty_like(y)
        BLOCK_C = triton.next_power_of_2(cols)
        _relu_softmax_kernel[(rows,)](y, out, cols, y.stride(0), BLOCK=BLOCK_C, num_warps=8)
        return out
