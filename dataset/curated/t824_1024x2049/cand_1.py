import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 824
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out = num / denom
    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fused matmul + bias via cuBLAS (tensor cores on A100)
        y = torch.addmm(self.b1, x, self.W0)
        if not y.is_cuda:
            return torch.softmax(y, dim=-1)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _softmax_kernel[(m,)](
            y, out,
            y.stride(0), out.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
