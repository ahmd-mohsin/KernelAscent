import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 668
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _softmax_rms_kernel(X, W, Y, D: tl.constexpr, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulate, matching PyTorch CUDA half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # cast to fp16 (softmax output dtype), then back to fp32 for RMS norm
    sm16 = sm.to(tl.float16)
    xf = sm16.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    out = (xf * inv).to(tl.float16) * w  # fp16 multiply, matching reference

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xs = torch.softmax(x, dim=-1)
            _xf = xs.float()
            return (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xs.dtype) * self.rms1_w

        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _softmax_rms_kernel[(rows,)](
            x, self.rms1_w, y, d, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return y
