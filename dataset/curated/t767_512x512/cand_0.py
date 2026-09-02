import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 767
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(X, W, Out, N, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, output rounded to fp16 like torch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x16 = (e / s).to(tl.float16)

    # RMSNorm in fp32 from fp16 values
    xf = x16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    xn16 = (xf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float16)
    y16 = xn16 * w  # fp16 multiply (matches half*half elementwise)

    # GELU (exact, computed in fp32, cast to fp16 like torch half kernel)
    yf = y16.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # softmax 2 (fp32 accumulation)
    gf = tl.where(mask, g16.to(tl.float32), float('-inf'))
    m2 = tl.max(gf, axis=0)
    e2 = tl.exp(gf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = ((e2 / s2).to(tl.float16).to(tl.float32) * 1.2619).to(tl.float16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_kernel[(Mrows,)](
            h, self.rms2_w, out, N,
            h.stride(0), out.stride(0),
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
