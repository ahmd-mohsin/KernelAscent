import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 362
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_softmax_scale_bias_softmax(
    Y_ptr, B_ptr, OUT_ptr,
    stride_y, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load matmul output row (bf16 -> fp32) ----
    y = tl.load(Y_ptr + row * stride_y + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accumulation, like PyTorch's bf16 softmax) ----
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p = e1 / s1
    # round to bf16 to match PyTorch's intermediate output dtype
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- scale by 1.2039 (fp32 math, round to bf16, matching x * scalar) ----
    p = (p * 1.2039).to(tl.bfloat16).to(tl.float32)

    # ---- add bias (fp32 math, round to bf16) ----
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (p + b).to(tl.bfloat16).to(tl.float32)

    # ---- softmax #2 ----
    z = tl.where(mask, z, float('-inf'))
    m2 = tl.max(z, axis=0)
    e2 = tl.exp(z - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(OUT_ptr + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores (bf16)
        y = torch.matmul(x, self.W0)

        rows, cols = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_softmax_scale_bias_softmax[(rows,)](
            y, self.b3, out,
            y.stride(0), out.stride(0),
            N=cols,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
