import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 380
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, G, B, W, Out,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    h = tl.maximum(x, 0.0)

    # layer norm (fp32 math, bf16 output like PyTorch)
    mean = tl.sum(h, axis=0) / N
    diff = tl.where(mask, h - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (h - mean) * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # relu
    z = tl.maximum(y, 0.0)

    # softmax (fp32 accumulation, bf16 output)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # rms norm
    ms = tl.sum(p * p, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    pn = (p * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (pn * w).to(tl.bfloat16)

    tl.store(Out + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        _fused_kernel[(m,)](
            x, self.ln2_g, self.ln2_b, self.rms5_w, out,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
