import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 892
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W1, W4, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, erf-based; computed in fp32, rounded to fp16 like torch) ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # ---- RMSNorm 1 (stats in fp32, cast to fp16, then fp16 multiply by weight) ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    xh = (x * r).to(tl.float16) * w1  # fp16 arithmetic, matches reference
    x = xh.to(tl.float32)

    # ---- GELU ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 accumulate, output rounded to fp16) ----
    xm = tl.where(mask, x, float('-inf'))
    m = tl.max(xm, axis=0)
    e = tl.math.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.float16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms2 = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0)
    y = (x * r2).to(tl.float16) * w4

    tl.store(Y + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        rows = xc.shape[0]
        y = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            xc, self.rms1_w, self.rms4_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
