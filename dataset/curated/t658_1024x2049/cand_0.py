import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 658
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _bias_softmax_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + row * stride_xm + offs, mask=mask, other=-float('inf')).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_ym + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0  # cuBLAS GEMM (bf16, tensor cores)
        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_softmax_kernel[(m,)](
            z, self.b1, out,
            z.stride(0), out.stride(0),
            n, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
