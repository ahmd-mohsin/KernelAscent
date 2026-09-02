import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 790
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _bias3_softmax_kernel(
    X, B1, B2, B3, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)

    # replicate sequential bf16 rounding of x + b1 + b2 + b3
    x = (x + b1).to(tl.bfloat16)
    x = (x + b2).to(tl.bfloat16)
    x = (x + b3).to(tl.bfloat16)

    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16)

    tl.store(Out + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (fp32 accumulation), matches reference matmul
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _bias3_softmax_kernel[(Mrows,)](
            h, self.b1, self.b2, self.b3, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )
        return out
