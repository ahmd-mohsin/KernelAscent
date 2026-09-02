import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 73
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_kernel(X, Wrms, G, B, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # load row (fp16 -> fp32)
    x = tl.load(X + base + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 internal, round to fp16) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    h = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- gelu (erf-based, fp32 opmath, round to fp16) ----
    g = h * 0.5 * (1.0 + tl.math.erf(h * 0.7071067811865476))
    h = g.to(tl.float16).to(tl.float32)

    # ---- softmax 2 ----
    m2 = tl.max(tl.where(mask, h, float('-inf')), 0)
    e2 = tl.exp(h - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    h = (e2 / s2).to(tl.float16).to(tl.float32)

    # ---- rmsnorm (explicit fp32, round to fp16, then * weight) ----
    ms = tl.sum(h * h, 0) / D
    r = h * (1.0 / tl.sqrt(ms + 1e-6))
    h = r.to(tl.float16).to(tl.float32)
    w = tl.load(Wrms + offs, mask=mask, other=0.0).to(tl.float32)
    h = (h * w).to(tl.float16).to(tl.float32)

    # ---- layernorm (fp32 stats, eps=1e-5) ----
    mu = tl.sum(tl.where(mask, h, 0.0), 0) / D
    d = tl.where(mask, h - mu, 0.0)
    var = tl.sum(d * d, 0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    gg = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * gg + bb
    tl.store(Y + base + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        x = torch.matmul(x, self.W0)
        x = x.contiguous()
        d = x.shape[-1]
        rows = x.numel() // d
        y = torch.empty_like(x)
        _fused_kernel[(rows,)](
            x, self.rms4_w, self.ln5_g, self.ln5_b, y,
            D=d, BLOCK=triton.next_power_of_2(d),
            num_warps=4,
        )
        return y
