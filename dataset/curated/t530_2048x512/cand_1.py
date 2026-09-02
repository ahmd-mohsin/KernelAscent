import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 530
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_softmax_rms_rms_ln(
    X, B1, W3, W4, G5, B5, Out,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16) and add bias in fp16 (matches x + b1 in half)
    x = tl.load(X + row * stride + offs, mask=mask, other=0.0)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    x = x + b1

    # softmax: compute in fp32, output fp16 (matches torch.softmax on half)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, 0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = (e / s).to(tl.float16)

    # RMSNorm #1: fp32 stats, cast to fp16, then fp16 multiply by weight
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    z = (yf * r).to(tl.float16) * w3

    # RMSNorm #2
    zf = z.to(tl.float32)
    ms2 = tl.sum(zf * zf, 0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0)
    u = (zf * r2).to(tl.float16) * w4

    # LayerNorm: fp32 internal math (matches F.layer_norm on half)
    uf = u.to(tl.float32)
    mean = tl.sum(tl.where(mask, uf, 0.0), 0) / N
    d = tl.where(mask, uf - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G5 + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((uf - mean) * rstd * g + b5).to(tl.float16)

    tl.store(Out + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM with tensor cores
        h = x @ self.W0  # (M, 4096) fp16

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_rms_rms_ln[(Mrows,)](
            h, self.b1, self.rms3_w, self.rms4_w, self.ln5_g, self.ln5_b, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
