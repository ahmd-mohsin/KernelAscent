import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 413
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    X_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    # relu
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))
    # softmax (fp32 accumulation, matching PyTorch CUDA softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom
    tl.store(Y_ptr + row * stride_y + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmuls (same numerics as reference)
        h = x @ self.W0
        h = h @ self.W1
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_softmax_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
