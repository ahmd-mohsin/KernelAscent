import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 342
M, D, DT = 512, 1024, torch.float16


@triton.jit
def double_softmax_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # first softmax (fp32 compute, round to fp16 to match reference intermediate)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16).to(tl.float32)

    # second softmax
    m2 = tl.max(tl.where(mask, p1, -float('inf')), axis=0)
    e2 = tl.exp(p1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    tl.store(Y + row * stride_ym + cols, p2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0  # cuBLAS fp16 GEMM (M, 512)
        if not z.is_cuda:
            z = torch.softmax(z, dim=-1)
            return torch.softmax(z, dim=-1)
        z = z.contiguous()
        Mrows, N = z.shape
        y = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(N)
        double_softmax_kernel[(Mrows,)](
            z, y, N,
            z.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8 if BLOCK_N >= 512 else 4,
        )
        return y
