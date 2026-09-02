import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 42
M, D, DT = 512, 512, torch.float16


@triton.jit
def _softmax_rms_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax with float accumulation)
    row_max = tl.max(xf, axis=0)
    e = tl.math.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # round to fp16 (softmax output dtype), then RMSNorm in fp32
    sm16 = sm.to(tl.float16)
    yf = sm16.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)

    z16 = (yf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # half*half elementwise in PyTorch uses fp32 opmath, rounds to half
    z = (z16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    out = (z.to(tl.float32) * 1.1093).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _softmax_rms_kernel[(m,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
