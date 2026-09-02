import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 889
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(X, Y, N, stride_x, stride_y, SCALE: tl.constexpr,
                              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # scale (round to fp16 to match reference intermediate)
    xf = x.to(tl.float32) * SCALE
    xf = xf.to(tl.float16).to(tl.float32)

    # exact erf-based GELU in fp32, then round to fp16
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32
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
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_gelu_softmax[(Mrows,)](
            h, out, N, h.stride(0), out.stride(0),
            SCALE=1.1778, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )
        return out
