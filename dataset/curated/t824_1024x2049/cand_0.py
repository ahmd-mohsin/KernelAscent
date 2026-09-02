import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 824
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _bias_softmax_kernel(
    X, B, Out,
    stride_xm,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out + row * stride_xm + cols, y.to(Out.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via tensor cores
        h = torch.matmul(x, self.W0)

        if not h.is_cuda:
            return torch.softmax(h + self.b1, dim=-1)

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _bias_softmax_kernel[(Mrows,)](
            h, self.b1, out,
            h.stride(0),
            N, BLOCK_N,
            num_warps=8,
        )
        return out
