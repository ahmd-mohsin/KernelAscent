import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 664
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_epilogue(X, B, W, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)

    # bias add (half + half computed at fp32 then rounded to fp16, matching PyTorch)
    xf = x.to(tl.float32) + b.to(tl.float32)
    xh = xf.to(tl.float16)

    # exact GELU (erf), computed at fp32 then rounded to fp16 (matches PyTorch half gelu)
    g = xh.to(tl.float32)
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    gh = g.to(tl.float16)

    # softmax with fp32 accumulation, output rounded to fp16
    s = tl.where(mask, gh.to(tl.float32), float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    smh = sm.to(tl.float16)

    # RMS norm: fp32 math, cast to fp16, then multiply by weight
    xf2 = smh.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf2 * xf2, 0.0), axis=0) / N
    rms = xf2 * tl.math.rsqrt(ms + 1e-6)
    rmsh = rms.to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    out = (rmsh.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # ReLU
    zero = tl.zeros_like(out)
    out = tl.maximum(out, zero)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS GEMM (tensor cores)
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue[(Mrows,)](
            h, self.b1, self.rms4_w, y,
            h.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
