import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 143
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _softmax_scale_bias_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_xm + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    out = p * SCALE + b
    tl.store(Y + row * stride_ym + offs, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0
        m, n = z.shape
        y = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(n)
        _softmax_scale_bias_kernel[(m,)](
            z, self.b3, y,
            z.stride(0), y.stride(0),
            n, 1.2932, BLOCK,
            num_warps=8,
        )
        return y
