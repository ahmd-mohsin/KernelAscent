import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 86
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, B0, W, OUT,
    stride_xm, stride_om,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0  (fp16 elementwise add: compute fp32, round to fp16)
    x = (x + b0).to(tl.float16).to(tl.float32)

    # softmax 1 (fp32 accumulation, output rounded to fp16)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # RMS norm (fp32), cast to fp16, multiply by weight (fp16 result)
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.float16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xn * w).to(tl.float16).to(tl.float32)

    # softmax 2
    x = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x, axis=0)
    e2 = tl.exp(x - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y = (e2 / s2).to(tl.float16)

    tl.store(OUT + row * stride_om + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        Mrows, Dcols = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x2, self.b0, self.rms2_w, out,
            x2.stride(0), out.stride(0),
            D_=Dcols, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
