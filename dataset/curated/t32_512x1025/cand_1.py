import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 32
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_kernel(X, G, B, B4, W5, OUT,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, cast to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    ln = (d * rstd) * g + b
    ln = ln.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, fp16 output) ----
    ln_m = tl.where(mask, ln, float('-inf'))
    mx = tl.max(ln_m, axis=0)
    e = tl.where(mask, tl.exp(ln - mx), 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # ---- GELU exact (erf), fp32 math, cast fp16 ----
    gel = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    gel = gel.to(tl.float16).to(tl.float32)

    # ---- Add bias ----
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (gel + b4).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32), cast fp16, then multiply by weight ----
    ms = tl.sum(tl.where(mask, xb * xb, 0.0), axis=0) / N
    rr = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xb * rr).to(tl.float16).to(tl.float32)
    w5 = tl.load(W5 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w5).to(tl.float16)

    tl.store(OUT + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        _fused_kernel[(Mrows,)](
            x, self.ln1_g, self.ln1_b, self.b4, self.rms5_w, out,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=4,
        )
        return out
