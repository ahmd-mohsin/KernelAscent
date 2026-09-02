import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 310
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_gelu_rms_softmax(X, W, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed then rounded to bf16 like reference
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(g * g, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    h = (g * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    h = (h * w).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    h = tl.where(mask, h, float('-inf'))
    m = tl.max(h, axis=0)
    e = tl.exp(h - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _fused_gelu_rms_softmax[(Mrows,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y
