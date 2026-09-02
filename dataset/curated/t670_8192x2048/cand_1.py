import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 670
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_relu_bias_softmax(
    X, B2, B3, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)

    # relu then add biases (in bf16 to match reference numerics, then softmax in fp32)
    x = tl.maximum(x, 0.0)
    v = x.to(tl.bfloat16) + b2.to(tl.bfloat16)
    v = v + b3.to(tl.bfloat16)
    v = v.to(tl.float32)

    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_relu_bias_softmax[(m,)](
            y, self.b2, self.b3, out,
            y.stride(0), out.stride(0),
            n, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
