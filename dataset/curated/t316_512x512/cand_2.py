import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 316
M, D, DT = 512, 512, torch.float16


@triton.jit
def _bias_softmax_scale_kernel(
    X, B, Out,
    N, stride_xm, stride_om,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float("-inf")).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom * SCALE

    tl.store(Out + row * stride_om + cols, y.to(Out.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        if not y.is_cuda:
            y = y + self.b1
            y = torch.softmax(y, dim=-1)
            return y * 1.018

        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _bias_softmax_scale_kernel[(m,)](
            y, self.b1, out,
            n, y.stride(0), out.stride(0),
            SCALE=1.018,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
