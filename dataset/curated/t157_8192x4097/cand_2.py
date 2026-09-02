import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 157
M, D, DT = 8192, 4097, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    S0, S1,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf'))

    # Replicate PyTorch fp16 scalar-mul semantics: compute in fp32, round to fp16 each step
    x = (x.to(tl.float32) * S0).to(tl.float16)
    x = (x.to(tl.float32) * S1).to(tl.float16)

    # Softmax with fp32 accumulation (as PyTorch does for half inputs)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM (same as reference matmul)
        h = x @ self.W0

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _scale_softmax_kernel[(Mrows,)](
            h, out,
            N, h.stride(0), out.stride(0),
            1.0453, 1.0092,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
