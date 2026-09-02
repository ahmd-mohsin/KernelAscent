import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 790
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _bias3_softmax_kernel(
    X_ptr, B1_ptr, B2_ptr, B3_ptr, Out_ptr,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0)

    # Replicate the sequential bf16 bias additions of the reference.
    x = x + b1
    x = x + b2
    x = x + b3

    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * stride_om + offs, out.to(Out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # High-performance cuBLAS matmul (fp32 accumulate on tensor cores).
        y = torch.matmul(x, self.W0)

        if not y.is_cuda:
            y = y + self.b1
            y = y + self.b2
            y = y + self.b3
            return torch.softmax(y, dim=-1)

        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _bias3_softmax_kernel[(m,)](
            y, self.b1, self.b2, self.b3, out,
            n, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
