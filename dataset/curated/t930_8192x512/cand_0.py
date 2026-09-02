import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 930
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, B_ptr, Out_ptr,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)

    # scale (round to bf16 to match reference elementwise rounding)
    x = (x * 1.0855).to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # + bias
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16).to(tl.float32)

    # relu
    y = tl.maximum(y, 0.0)

    # softmax (fp32 accumulation)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * stride_om + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        grid = (m,)
        _fused_epilogue_kernel[grid](
            h, self.b4, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
