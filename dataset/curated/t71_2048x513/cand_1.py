import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 71
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _softmax_bias_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 accumulation, like PyTorch on fp16 inputs)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(tl.where(mask, e1, 0.0), axis=0)
    p1 = e1 / s1

    # round to fp16 (matches materialized fp16 tensor), add bias in fp16
    p1_h = p1.to(tl.float16)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    z_h = p1_h + b.to(tl.float16)
    z = z_h.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))

    # second softmax
    m2 = tl.max(z, axis=0)
    e2 = tl.exp(z - m2)
    s2 = tl.sum(tl.where(mask, e2, 0.0), axis=0)
    p2 = e2 / s2

    tl.store(Y_ptr + row * stride_ym + cols, p2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_bias_softmax_kernel[(m,)](
            h, self.b2, y,
            n, h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
