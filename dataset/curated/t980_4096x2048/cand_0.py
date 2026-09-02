import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 980
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _relu_scale_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    x = tl.maximum(x, 0.0)
    x = x * scale
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y_ptr + row * stride_y + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _relu_scale_softmax_kernel[(Mrows,)](
            h, out, N, h.stride(0), out.stride(0), 1.4048,
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        return out
