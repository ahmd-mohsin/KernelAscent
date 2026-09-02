import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 869
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_kernel(
    X, W2, W4, Y,
    D, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load row, scale + relu (opmath float, round to fp16 like PyTorch)
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * 1.0531
    x = tl.maximum(x, 0.0)
    x = x.to(tl.float16).to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * r).to(tl.float16).to(tl.float32) * w2
    y = y.to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, output rounded to fp16)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm 2
    ms2 = tl.sum(p * p, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((p * r2).to(tl.float16).to(tl.float32) * w4).to(tl.float16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.0531
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2d, self.rms2_w, self.rms4_w, y,
            d, x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
            num_stages=1,
        )
        return y.view(orig_shape)
