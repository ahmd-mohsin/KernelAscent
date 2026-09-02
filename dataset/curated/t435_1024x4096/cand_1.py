import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 435
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


@triton.jit
def _ln_rms_gelu_kernel(X, G, B, W, Y, N, ln_eps, rms_eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, output rounded to fp16 like PyTorch)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + ln_eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln16 = (xc * rstd * g + b).to(tl.float16)

    # RMSNorm: computed in fp32 on the fp16 LN output, cast to fp16, then * w (fp16 mul)
    lnf = ln16.to(tl.float32)
    ms = tl.sum(tl.where(mask, lnf * lnf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + rms_eps)
    y16 = (lnf * rrms).to(tl.float16)
    w16 = tl.load(W + cols, mask=mask, other=0.0)
    z16 = y16 * w16  # fp16 multiply, matching reference

    # GELU (exact, erf-based) computed in fp32, cast back to fp16 (matches PyTorch half gelu)
    zf = z16.to(tl.float32)
    out = 0.5 * zf * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # (M, 1024)
        x = x.contiguous()
        rows, N1 = x.shape
        sm = torch.empty_like(x)
        _softmax_kernel[(rows,)](x, sm, N1, BLOCK=triton.next_power_of_2(N1),
                                 num_warps=8)
        h = sm @ self.W2  # (M, 512)
        h = h.contiguous()
        rows2, N2 = h.shape
        out = torch.empty_like(h)
        _ln_rms_gelu_kernel[(rows2,)](h, self.ln3_g, self.ln3_b, self.rms4_w,
                                      out, N2, 1e-5, 1e-6,
                                      BLOCK=triton.next_power_of_2(N2),
                                      num_warps=4)
        return out
