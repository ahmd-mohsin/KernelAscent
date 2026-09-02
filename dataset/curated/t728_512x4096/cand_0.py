import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 728
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _softmax_relu_kernel(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    y = tl.maximum(y, 0.0)
    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _softmax_relu_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            n, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
