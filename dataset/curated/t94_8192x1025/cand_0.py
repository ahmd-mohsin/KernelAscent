import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 94
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N,
    stride_xm,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # exact gelu (erf), computed in fp32 then rounded to bf16 (match torch)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu (identity on nonneg, kept for fidelity)
    g = tl.maximum(g, 0.0)

    # + bias, rounded to bf16 (match torch bf16 add)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y, axis=0)
    e = tl.exp(y - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (bf16, tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_act_softmax_kernel[(m,)](
            h, self.b4, out,
            n,
            h.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
