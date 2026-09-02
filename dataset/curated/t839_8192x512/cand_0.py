import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 839
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_
    x = tl.load(X + row * D_ + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax #1 (computed in fp32, rounded to fp16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # softmax #2
    x = x.to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # RMSNorm in fp32, cast to fp16, multiply by fp16 weight
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    y = xn * w
    tl.store(Y + row * D_ + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x, self.rms2_w, y, d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
