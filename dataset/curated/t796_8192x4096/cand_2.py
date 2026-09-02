import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 796
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(X, W, Y, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = e / s
    # round softmax output to bf16 (matches torch.softmax output dtype), then upcast
    pb = p.to(tl.bfloat16)
    pf = pb.to(tl.float32)
    ms = tl.sum(pf * pf, 0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (pf * r).to(tl.bfloat16) * w
    tl.store(Y + row * D + offs, y, mask=mask)


@triton.jit
def _softmax_ln_kernel(X, G, B, Y, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = e / s
    # round softmax output to bf16 (matches torch.softmax output dtype), then upcast
    pb = p.to(tl.bfloat16)
    pf = pb.to(tl.float32)
    mu = tl.sum(pf, 0) / D
    diff = tl.where(mask, pf - mu, 0.0)
    var = tl.sum(diff * diff, 0) / D
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (pf - mu) * inv * g + b
    tl.store(Y + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = x @ self.W2
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        x = x.contiguous()
        Mrows, Dcols = x.shape
        BLOCK = triton.next_power_of_2(Dcols)

        t = torch.empty_like(x)
        _softmax_rms_kernel[(Mrows,)](
            x, self.rms1_w, t, Dcols, BLOCK=BLOCK, num_warps=16
        )

        h = t @ self.W2  # tensor-core bf16 GEMM

        out = torch.empty_like(h)
        _softmax_ln_kernel[(Mrows,)](
            h, self.ln4_g, self.ln4_b, out, Dcols, BLOCK=BLOCK, num_warps=16
        )
        return out
