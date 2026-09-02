import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 993
M, D, DT = 512, 513, torch.bfloat16


@triton.jit
def _softmax_scale_bias_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    sm = num / denom
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = sm * SCALE + b
    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_scale_bias_kernel[(m,)](
            h, self.b3, y,
            h.stride(0), y.stride(0),
            N=n, SCALE=1.1294, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
