import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 569
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_bias_softmax_kernel(
    X, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # bias add in bf16 to match reference numerics (x + b0 done in bf16)
    xb = (x.to(tl.bfloat16) + b.to(tl.bfloat16)).to(tl.float32)
    xb = tl.where(mask, xb, float('-inf'))

    m = tl.max(xb, axis=0)
    e = tl.exp(xb - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_bias_softmax_kernel[(M_,)](
            x, self.b0, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
