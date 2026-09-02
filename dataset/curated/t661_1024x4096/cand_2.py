import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 661
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_gelu2_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulate, like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (softmax output dtype), then gelu in fp32 opmath
    p16 = p.to(tl.float16)
    v = p16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = v * 0.5 * (1.0 + tl.math.erf(v * INV_SQRT2))
    g16 = g.to(tl.float16)

    v2 = g16.to(tl.float32)
    g2 = v2 * 0.5 * (1.0 + tl.math.erf(v2 * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, g2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_gelu2_kernel[(m,)](
            x, y, n, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
