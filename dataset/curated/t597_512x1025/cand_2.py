import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 597
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 compute, output rounded to fp16 like PyTorch)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # gelu (exact erf, fp32 opmath, round to fp16)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = g.to(tl.float16).to(tl.float32)

    # scale (fp32 opmath, round to fp16)
    x = (x * 1.2534).to(tl.float16).to(tl.float32)

    # softmax 2
    x = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x, 0)
    e2 = tl.exp(x - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    x = (e2 / s2).to(tl.float16)

    # relu
    x = tl.maximum(x, 0.0)

    tl.store(Y + row * stride_y + cols, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            h, out, N, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
