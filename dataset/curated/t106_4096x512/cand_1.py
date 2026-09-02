import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 106
M, D, DT = 4096, 512, torch.bfloat16


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
    e = tl.exp(x - row_max)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bias_add_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    tl.store(Y + row * stride_ym + cols, x + b, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = x @ self.W1
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, d = x.shape

        # Fused bias add (elementwise, single pass)
        xb = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(d)
        _bias_add_kernel[(m,)](
            x, self.b0, xb,
            x.stride(0), xb.stride(0),
            d,
            BLOCK_N=BLOCK_D,
            num_warps=4,
        )

        # cuBLAS GEMM (tensor cores, bf16 with fp32 accumulate)
        y = xb @ self.W1

        # Fused Triton softmax over the last dim
        n = y.shape[-1]
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
