import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 811
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def fused_bias_softmax_kernel(
    X, B0, B1, OUT,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)

    # replicate reference rounding: (x + b0) -> bf16, (+ b1) -> bf16
    t = (x + b0).to(tl.bfloat16)
    t = (t + b1).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch internal upcast)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    row_max = tl.max(tf, axis=0)
    e = tl.exp(tf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(OUT + row * stride_om + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        fused_bias_softmax_kernel[(m,)](
            x, self.b0, self.b1, out,
            x.stride(0), out.stride(0),
            N=n, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
