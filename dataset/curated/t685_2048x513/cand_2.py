import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 685
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _softmax_scale_kernel(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    denom = tl.sum(e, axis=0)
    y = (e / denom) * SCALE
    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_scale_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            n, 1.0138, BLOCK_N,
            num_warps=8,
        )
        return out
