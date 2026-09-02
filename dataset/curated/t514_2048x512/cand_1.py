import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 514
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_softmax_double_rms(
    X, W2, W3, Out,
    N,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    # load row (fp16 -> fp32)
    x = tl.load(X + base + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    p = e / s

    # cast to fp16 (matches torch softmax output dtype), then upcast for RMS
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)
    pf = tl.where(mask, pf, 0.0)

    # first RMSNorm
    ms1 = tl.sum(pf * pf, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + EPS)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    y1 = (pf * r1).to(tl.float16) * w2  # fp16 multiply, as in reference

    # second RMSNorm
    y1f = y1.to(tl.float32)
    y1f = tl.where(mask, y1f, 0.0)
    ms2 = tl.sum(y1f * y1f, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + EPS)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    y2 = (y1f * r2).to(tl.float16) * w3

    tl.store(Out + base + offs, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_double_rms[(Mrows,)](
            h, self.rms2_w, self.rms3_w, out,
            N, EPS=1e-6, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
