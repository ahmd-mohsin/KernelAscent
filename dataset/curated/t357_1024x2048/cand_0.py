import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 357
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_softmax_rms_gelu(
    X, W, Y,
    stride_xm, stride_ym,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch fp16 softmax which uses fp32 opmath)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # cast to fp16 (softmax output dtype), then RMSNorm in fp32
    sm16 = sm.to(tl.float16)
    xf = sm16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D_
    rs = 1.0 / tl.sqrt(ms + 1e-6)
    xn16 = (xf * rs).to(tl.float16)

    # multiply by weight in fp16 (matches fp16 * fp16 elementwise)
    w = tl.load(W + offs, mask=mask, other=0.0)
    h16 = xn16 * w

    # GELU (exact, erf) with fp32 opmath, matching PyTorch's fp16 gelu
    h = h16.to(tl.float32)
    g = 0.5 * h * (1.0 + tl.math.erf(h * 0.7071067811865476))
    out = g.to(tl.float16)

    tl.store(Y + row * stride_ym + offs, out, mask=mask)


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
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_rms_gelu[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
