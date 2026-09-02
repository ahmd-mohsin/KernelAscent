import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 903
M, D, DT = 512, 513, torch.float16


@triton.jit
def _fused_act_softmax(X, Y, N, stride_x, stride_y, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu 1 (exact, fp32 math, round to fp16 to match PyTorch intermediates)
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # gelu 2
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax (fp32 accumulation)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_act_softmax[(m,)](
            h, y, n, h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N, num_warps=8,
        )
        return y
