import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 135
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_rows(X, W2, W4, Y, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)  # fp16
    # relu (exact in fp16)
    x = tl.where(x > 0, x, 0.0)
    xf = x.to(tl.float32)

    # RMSNorm 1 (fp32 mean, round to fp16, then fp16-weight mul in fp32, round to fp16)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    t = (xf * inv).to(tl.float16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    t = (t * w2).to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, output rounded to fp16)
    t_m = tl.where(mask, t, float('-inf'))
    mx = tl.max(t_m, axis=0)
    e = tl.exp(t_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm 2
    ms2 = tl.sum(sm * sm, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    t2 = (sm * inv2).to(tl.float16).to(tl.float32)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.float32)
    t2 = (t2 * w4).to(tl.float16)

    # relu
    out = tl.where(t2 > 0, t2, 0.0)
    tl.store(Y + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rows[(Mrows,)](
            h, self.rms2_w, self.rms4_w, y,
            N, h.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return y
