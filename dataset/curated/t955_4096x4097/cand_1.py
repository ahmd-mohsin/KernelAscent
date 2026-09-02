import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 955
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    # fused ReLU
    x = tl.maximum(x, 0.0)
    # mask out-of-bounds lanes for softmax
    x = tl.where(mask, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_ym + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x @ self.W1
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        # ReLU (elementwise, memory-bound; cheap) then cuBLAS GEMM (fast, matches reference numerics)
        x = torch.relu(x)
        h = torch.matmul(x, self.W1)
        if not h.is_contiguous():
            h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _relu_softmax_kernel[(Mrows,)](
            h, out,
            N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
