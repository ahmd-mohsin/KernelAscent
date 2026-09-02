import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 525
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_softmax_rms_bias(
    X, W, B, Y,
    stride_xm, stride_ym,
    D_: tl.constexpr, BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax accumulate behavior)
    xmax = tl.max(x, axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (softmax output dtype), then upcast for RMS norm
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)

    ms = tl.sum(pf * pf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + EPS)
    n16 = (pf * inv).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    y = n16 * w + b

    tl.store(Y + row * stride_ym + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x + self.b2

        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_rms_bias[(m,)](
            x, self.rms1_w, self.b2, y,
            x.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK, EPS=1e-6,
            num_warps=4,
        )
        return y
