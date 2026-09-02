import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 328
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W0, W3, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 0
    ms = tl.sum(x * x, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    h = ((x * inv).to(tl.float16) * w0).to(tl.float32)

    # softmax 1 (fp32 compute, fp16 round like PyTorch output dtype)
    h = tl.where(mask, h, float('-inf'))
    m1 = tl.max(h, axis=0)
    e1 = tl.exp(h - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16).to(tl.float32)

    # softmax 2
    p1m = tl.where(mask, p1, float('-inf'))
    m2 = tl.max(p1m, axis=0)
    e2 = tl.exp(p1m - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.float16).to(tl.float32)

    # RMSNorm 3
    ms2 = tl.sum(p2 * p2, axis=0) / D_
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    o = (p2 * inv2).to(tl.float16) * w3

    # ReLU
    o = tl.maximum(o, tl.zeros_like(o))

    tl.store(Y + row * D_ + offs, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return torch.relu(x)

        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms0_w, self.rms3_w, y,
            D_=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
