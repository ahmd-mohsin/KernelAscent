import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 157
M, D, DT = 8192, 4097, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # replicate fp16 rounding of the two successive scalar multiplies
    x = (x * S1).to(tl.float16)
    x = (x * S2).to(tl.float16)

    xf = x.to(tl.float32)
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N_)
        _scale_softmax_kernel[(M_,)](
            x, y,
            x.stride(0), y.stride(0),
            N_,
            1.0453, 1.0092,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
