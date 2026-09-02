import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 560
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X_ptr, B_ptr, Y_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, computed in fp32, rounded to fp16 like PyTorch half kernel)
    xf = x.to(tl.float32)
    g1 = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = g1.to(tl.float16)

    # scale in fp16
    x = (x * tl.full((), 1.2123, tl.float16)).to(tl.float16)

    # relu
    x = tl.maximum(x, tl.zeros_like(x))

    # gelu again
    xf = x.to(tl.float32)
    g2 = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = g2.to(tl.float16)

    # add bias in fp16
    x = (x + b).to(tl.float16)

    # softmax in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x, self.b4, y, d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
