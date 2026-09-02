import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 238
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _bias_softmax_kernel(
    X, B, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp16 (matches x + b1 in reference), then softmax in fp32
    x = x + b
    xf = x.to(tl.float32)

    row_max = tl.max(xf, axis=0)
    xf = xf - row_max
    num = tl.exp(xf)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Out + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 tensor-core matmul
        y = torch.matmul(x, self.W0)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_softmax_kernel[(m,)](
            y, self.b1, out,
            y.stride(0), out.stride(0),
            n, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
