import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 839
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, stride_x, stride_y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (float accum, output cast to fp16 like PyTorch)
    m1 = tl.max(x, axis=0)
    e1 = tl.math.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = (e1 / s1).to(tl.float16)

    # softmax 2
    x2 = tl.where(mask, y1.to(tl.float32), float('-inf'))
    m2 = tl.max(x2, axis=0)
    e2 = tl.math.exp(x2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = (e2 / s2).to(tl.float16)

    # RMSNorm in fp32, cast to fp16, then fp16 multiply by weight
    xf = y2.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    rstd = tl.math.rsqrt(ms + 1e-6)
    normed = (xf * rstd).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float16)
    out = normed * w
    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        w = self.rms2_w.to(device=x.device, dtype=torch.float16)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2d, w, y,
            x2d.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
