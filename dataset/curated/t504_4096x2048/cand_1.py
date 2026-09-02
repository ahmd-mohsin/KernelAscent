import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 504
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _double_softmax_kernel(
    X_ptr, Y_ptr,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 accumulate, like PyTorch does for fp16 inputs)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1

    # cast intermediate to fp16 (matches materialized fp16 tensor in reference)
    p1 = p1.to(tl.float16).to(tl.float32)
    p1 = tl.where(mask, p1, float('-inf'))

    # second softmax
    m2 = tl.max(p1, axis=0)
    e2 = tl.exp(p1 - m2)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    tl.store(Y_ptr + row * stride_ym + cols, p2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = z.shape
        y = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _double_softmax_kernel[(Mrows,)](
            z, y, N,
            z.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
