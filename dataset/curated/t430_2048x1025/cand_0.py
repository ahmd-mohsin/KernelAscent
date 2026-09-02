import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 430
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_row_kernel(X_ptr, W_ptr, Y_ptr, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact, computed in fp32 like PyTorch's half opmath, then rounded to fp16)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # softmax over the row (fp32 accumulate, fp16 output rounding)
    g = tl.where(mask, g, float("-inf"))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # gelu again
    g2 = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)

    # scale by 1.1055 (rounded to fp16 like the reference)
    g2 = (g2 * 1.1055).to(tl.float16).to(tl.float32)

    # rmsnorm in fp32, cast to fp16, multiply by fp16 weight
    ms = tl.sum(tl.where(mask, g2 * g2, 0.0), axis=0) / N
    r = (g2 * tl.math.rsqrt(ms + 1e-6)).to(tl.float16)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = r * w
    tl.store(Y_ptr + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_row_kernel[(rows,)](
            h, self.rms5_w, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
