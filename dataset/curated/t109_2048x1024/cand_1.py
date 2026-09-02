import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 109
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _softmax_bias_kernel(
    X_ptr, B_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = (num / denom).to(tl.float16)

    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    out = sm + b

    tl.store(Out_ptr + row * stride_xm + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        M_, N_ = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N_)
        _softmax_bias_kernel[(M_,)](
            y, self.b2, out,
            y.stride(0),
            N=N_,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
