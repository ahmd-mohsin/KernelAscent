import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 751
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, B1, W2, W3, G4, B4,
    D: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- softmax (fp32 accumulation, bf16 output rounding) ----
    x = tl.load(X + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    xb = (e / s).to(tl.bfloat16)

    # ---- + b1 (bf16 rounding) ----
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (xb.to(tl.float32) + b1).to(tl.bfloat16)

    # ---- rmsnorm 2 ----
    xf = xb.to(tl.float32)
    r = tl.math.rsqrt(tl.sum(xf * xf, 0) / D + EPS_RMS)
    xb = (xf * r).to(tl.bfloat16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (xb.to(tl.float32) * w2).to(tl.bfloat16)

    # ---- rmsnorm 3 ----
    xf = xb.to(tl.float32)
    r = tl.math.rsqrt(tl.sum(xf * xf, 0) / D + EPS_RMS)
    xb = (xf * r).to(tl.bfloat16)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (xb.to(tl.float32) * w3).to(tl.bfloat16)

    # ---- layernorm 4 ----
    xf = xb.to(tl.float32)
    mu = tl.sum(tl.where(mask, xf, 0.0), 0) / D
    d = tl.where(mask, xf - mu, 0.0)
    var = tl.sum(d * d, 0) / D
    inv = tl.math.rsqrt(var + EPS_LN)
    g = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b
    tl.store(Y + base + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)
        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        y = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(n_rows,)](
            xc, y, self.b1, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b,
            D=d, EPS_RMS=1e-6, EPS_LN=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        x = torch.softmax(x, dim=-1)
        x = x + self.b1
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x
