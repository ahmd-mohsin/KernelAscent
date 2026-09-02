import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 754
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_relu_bias_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    v = tl.maximum(x, 0.0) + b
    v = tl.where(mask, v, float('-inf'))

    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + offs, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_bias_softmax[(Mrows,)](
            x, self.b1, y,
            x.stride(0), y.stride(0),
            N, BLOCK,
            num_warps=num_warps,
        )
        return y
