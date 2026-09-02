import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 494
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_norms_gelu(X, OUT, W1, W2, G, B,
                      N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm 1 (fp32 math, round to bf16, bf16 weight multiply)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    t = (xf * r).to(tl.bfloat16)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    t = (t.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 2
    tf = t.to(tl.float32)
    ms2 = tl.sum(tf * tf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    t2 = (tf * r2).to(tl.bfloat16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    t2 = (t2.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm (fp32 opmath, eps=1e-5, biased variance)
    zf = t2.to(tl.float32)
    mu = tl.sum(zf, axis=0) / N
    diff = tl.where(mask, zf - mu, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (zf - mu) * inv * g + b
    y = y.to(tl.bfloat16)

    # GELU (exact erf, fp32 opmath)
    yf = y.to(tl.float32)
    out = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    tl.store(OUT + row * N + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        _fused_norms_gelu[(m,)](
            x, out, self.rms1_w, self.rms2_w, self.ln3_g, self.ln3_b,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
