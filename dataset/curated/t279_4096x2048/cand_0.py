import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 279
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # * 1.2475 (fp16 rounding as in reference)
    x = (x * 1.2475).to(tl.float16).to(tl.float32)
    # + bias (fp16 rounding)
    x = (x + b).to(tl.float16).to(tl.float32)
    # exact gelu (erf-based), computed in fp32 then rounded to fp16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = g.to(tl.float16).to(tl.float32)
    # * 1.2604 (fp16 rounding)
    x = (x * 1.2604).to(tl.float16).to(tl.float32)

    # softmax in fp32
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Out_ptr + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (same as reference)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_act_softmax_kernel[(m,)](
            h, self.b3, out,
            n, h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
