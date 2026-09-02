import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 474
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(X, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # relu (exact in fp16)
    x = tl.maximum(x, 0.0)

    # x * 1.0381 : computed in fp32 (opmath), rounded to fp16
    x = (x * 1.0381).to(tl.float16).to(tl.float32)

    # x + b2 : fp32 add, rounded to fp16
    x = (x + b).to(tl.float16).to(tl.float32)

    # exact GELU in fp32 (opmath), rounded to fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32, output fp16
    g_masked = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.relu(x)
            y = y * 1.0381
            y = y + self.b2
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        b = self.b2
        if b.device != x.device:
            b = b.to(x.device)
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x, b, y, n, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return y
