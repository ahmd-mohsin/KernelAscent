import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 323
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_kernel(X, W, B2, B4, Out,
                  stride_xm, stride_om,
                  D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)

    # multiply by weight in fp16 (matches reference)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float16)
    xh = xn * w

    # relu
    zero = tl.zeros_like(xh)
    xh = tl.where(xh > zero, xh, zero)

    # add b2 in fp16
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float16)
    xh = xh + b2

    # softmax: fp32 accumulation, fp16 output (matches PyTorch half softmax)
    xs = xh.to(tl.float32)
    xs = tl.where(mask, xs, float('-inf'))
    m = tl.max(xs, axis=0)
    e = tl.exp(xs - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # add b4 in fp16
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float16)
    out = sm + b4

    tl.store(Out + row * stride_om + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms0_w, self.b2, self.b4, out,
            x2.stride(0), out.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
