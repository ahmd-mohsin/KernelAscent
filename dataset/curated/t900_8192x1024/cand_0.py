import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 900
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_bias_gelu_softmax(
    X, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp16 (matches torch elementwise add on half)
    x = x + b

    # exact GELU in fp32 (matches torch's opmath for half), cast back to fp16
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g = g.to(tl.float16)

    # softmax with fp32 accumulation (matches torch softmax on half)
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, float("-inf"))
    row_max = tl.max(gf, axis=0)
    e = tl.exp(gf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_bias_gelu_softmax[(M_,)](
            x, self.b0, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
