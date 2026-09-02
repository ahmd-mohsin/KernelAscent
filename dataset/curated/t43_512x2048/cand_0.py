import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 43
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_epilogue(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf) -> round to fp16 like PyTorch intermediate tensor
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, fp16 output)
    gm = tl.where(mask, g, float("-inf"))
    m1 = tl.max(gm, axis=0)
    e1 = tl.where(mask, tl.exp(gm - m1), 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16).to(tl.float32)

    # relu: softmax output is nonnegative -> identity (exact)

    # second softmax
    pm = tl.where(mask, p1, float("-inf"))
    m2 = tl.max(pm, axis=0)
    e2 = tl.where(mask, tl.exp(pm - m2), 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.float16).to(tl.float32)

    # final gelu
    out = 0.5 * p2 * (1.0 + tl.math.erf(p2 * INV_SQRT2))
    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        if not h.is_cuda:
            h = F.gelu(h)
            h = torch.softmax(h, dim=-1)
            h = torch.relu(h)
            h = torch.softmax(h, dim=-1)
            return F.gelu(h)
        h = h.contiguous()
        rows, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue[(rows,)](
            h, y, n, h.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=8 if BLOCK >= 512 else 4,
        )
        return y
