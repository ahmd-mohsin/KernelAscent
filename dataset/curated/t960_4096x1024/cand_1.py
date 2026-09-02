import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 960
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, stride_x, stride_y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, output rounded to fp16 like torch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x16 = (e / s).to(tl.float16)

    # RMSNorm in fp32
    xf = x16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    xn16 = (xf * rrms).to(tl.float16)

    # multiply by weight in fp16
    w = tl.load(W + cols, mask=mask, other=0.0)
    xw16 = (xn16 * w).to(tl.float16)

    # GELU (erf), fp32 compute, rounded to fp16
    g = xw16.to(tl.float32)
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # softmax 2
    gf = tl.where(mask, g16.to(tl.float32), float('-inf'))
    m2 = tl.max(gf, axis=0)
    e2 = tl.exp(gf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        Mrows, Dcols = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x2, self.rms1_w, y,
            x2.stride(0), y.stride(0),
            D=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
