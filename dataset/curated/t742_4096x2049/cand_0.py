import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 742
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _bias_relu_softmax_kernel(
    X, B, Y,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    v = x + b
    v = tl.maximum(v, 0.0)
    v = tl.where(mask, v, float('-inf'))

    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_relu_softmax_kernel[(m,)](
            h, self.b1, y,
            n,
            h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
