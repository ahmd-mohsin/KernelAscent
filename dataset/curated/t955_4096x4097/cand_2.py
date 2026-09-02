import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 955
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _relu_kernel(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    zero = tl.zeros_like(x)
    tl.store(X + offs, tl.maximum(x, zero), mask=mask)


@triton.jit
def _relu_softmax_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # relu
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    out = num / den
    tl.store(Y + row * stride_y + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n = x.numel()
        # out-of-place relu to avoid mutating input
        xr = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _relu_oop_kernel[grid](x, xr, n, BLOCK=BLOCK)
        # cuBLAS GEMM (bf16, tensor cores)
        h = xr @ self.W1
        m, n_cols = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_N >= 512 else 4
        _relu_softmax_kernel[(m,)](
            h, out, n_cols, h.stride(0), out.stride(0),
            BLOCK=BLOCK_N, num_warps=num_warps,
        )
        return out


@triton.jit
def _relu_oop_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    zero = tl.zeros_like(x)
    tl.store(Y + offs, tl.maximum(x, zero), mask=mask)
