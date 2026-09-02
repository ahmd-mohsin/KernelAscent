import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 603
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _scale_bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x = x * SCALE + b
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Out_ptr + row * stride_om + cols, y.to(Out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS GEMM (fp16, uses tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _scale_bias_softmax_kernel[(m,)](
            h, self.b2, out,
            n, h.stride(0), out.stride(0),
            SCALE=1.2045,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
