import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 289
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _gelu_bias_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then cast back to fp16
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    y16 = g16 + b  # fp16 add, matching reference

    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))

    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK_N >= 4096 else 4
        _gelu_bias_softmax_kernel[(M_,)](
            h, self.b2, out,
            N_, h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        return out
