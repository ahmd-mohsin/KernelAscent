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

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # relu (fp16)
    x = tl.maximum(x, 0.0)
    # scale, mimic fp16 rounding
    x = (x.to(tl.float32) * 1.0381).to(tl.float16)
    # bias add (fp16)
    x = x + b
    # exact GELU, mimic fp16 rounding
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
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
        x = x.contiguous()
        rows, cols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2 = x.view(-1, cols)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        _fused_kernel[(x2.shape[0],)](
            x2, self.b2, y, cols,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view_as(x)
