import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 116
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _relu_double_softmax_kernel(
    X, Y,
    N, stride_x, stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # First softmax (fp32 accumulation like torch on fp16 input)
    x1 = tl.where(mask, x, float("-inf"))
    m1 = tl.max(x1, axis=0)
    e1 = tl.exp(x1 - m1)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1

    # Round to fp16 (intermediate tensor dtype), back to fp32
    p1 = p1.to(tl.float16).to(tl.float32)

    # Second softmax
    x2 = tl.where(mask, p1, float("-inf"))
    m2 = tl.max(x2, axis=0)
    e2 = tl.exp(x2 - m2)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    tl.store(Y + row * stride_y + offs, p2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 4096 else 4
        _relu_double_softmax_kernel[(Mrows,)](
            h, out, N, h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        return out
