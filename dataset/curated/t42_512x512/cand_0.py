import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 42
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_softmax_rms_kernel(
    X, W, OUT,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches torch fp16 softmax which accumulates in fp32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # cast to fp16 (reference softmax output dtype), then back to fp32 for RMS
    sm16 = sm.to(tl.float16)
    yf = sm16.to(tl.float32)

    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)

    z16 = (yf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    a = (z16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)  # fp16*fp16 product exact in fp32
    out = (a.to(tl.float32) * 1.1093).to(tl.float16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_rms_kernel[(m,)](
            y, self.rms2_w, out,
            y.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
