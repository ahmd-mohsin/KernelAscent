import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 171
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_kernel(X, W2, W4, Y, stride_x, stride_y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, output rounded to fp16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x16 = (e / s).to(tl.float16)

    # scale by 1.0869 (opmath fp32, round to fp16)
    x16 = (x16.to(tl.float32) * 1.0869).to(tl.float16)

    # rmsnorm 1
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x16 = (xf * r).to(tl.float16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    x16 = (x16.to(tl.float32) * w2).to(tl.float16)

    # relu
    x16 = tl.maximum(x16, tl.zeros_like(x16))

    # rmsnorm 2
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x16 = (xf * r).to(tl.float16)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    x16 = (x16.to(tl.float32) * w4).to(tl.float16)

    tl.store(Y + row * stride_y + offs, x16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback (reference path)
            x = torch.softmax(x, dim=-1)
            x = x * 1.0869
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.relu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms2_w, self.rms4_w, y,
            x.stride(0), y.stride(0),
            D_=Dcols, BLOCK=BLOCK,
            num_warps=8, num_stages=1,
        )
        return y
