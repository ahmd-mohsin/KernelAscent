import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 74
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    # relu
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))
    # softmax (numerically stable)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _relu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(Y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n = x.numel()
        xr = torch.empty_like(x)
        BLOCK = 1024
        _relu_kernel[(triton.cdiv(n, BLOCK),)](x, xr, n, BLOCK=BLOCK)

        # cuBLAS matmul (tensor cores)
        h = xr @ self.W1  # (M, 2048)

        m, ncols = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(ncols)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _relu_softmax_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            ncols,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
