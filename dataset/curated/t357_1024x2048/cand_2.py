import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 357
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, D_dim: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_dim

    x = tl.load(X + row * D_dim + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax which accumulates in fp32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # cast to fp16 then back to fp32 (matches x.float() after fp16 softmax output)
    xf = sm.to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / D_dim
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * r).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    y = xn * w  # fp16 multiply, matches PyTorch

    # gelu (erf-based) computed in fp32 (matches PyTorch opmath for half)
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    out = g.to(tl.float16)

    tl.store(Y + row * D_dim + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(x)

        x = x.contiguous()
        rows, d = x.shape[0], x.shape[-1]
        x2 = x.view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x2, self.rms1_w, y, d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view_as(x)
