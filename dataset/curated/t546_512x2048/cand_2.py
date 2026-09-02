import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 546
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_epilogue_softmax(
    X, B, Out,
    N,
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)                    # fp16

    # elementwise ops in fp16 to match reference precision
    c1 = tl.full((), 1.4852, tl.float16)
    c2 = tl.full((), 1.1826, tl.float16)
    y = x * c1
    y = tl.maximum(y, tl.full((), 0.0, tl.float16))
    y = y + b
    y = y * c2

    # softmax with fp32 accumulation (matches torch half softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float("-inf"))
    row_max = tl.max(yf, axis=0)
    num = tl.exp(yf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Out + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_epilogue_softmax[(m,)](
            h, self.b3, out,
            n,
            h.stride(0),
            out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
