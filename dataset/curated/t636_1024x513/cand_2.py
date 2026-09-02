import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 636
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_row_kernel(X_ptr, B2_ptr, RW_ptr, LG_ptr, LB_ptr, OUT_ptr,
                      N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    # ---- softmax (fp32 accumulation, bf16 output like torch) ----
    x = tl.load(X_ptr + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16)

    # ---- add bias (bf16 add) ----
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0)
    x = x + b2

    # ---- RMSNorm: fp32 compute, cast to bf16, then bf16 * weight ----
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    x = (xf * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16)
    rw = tl.load(RW_ptr + offs, mask=mask, other=0.0)
    x = x * rw

    # ---- LayerNorm (fp32 internal, affine in fp32, cast to bf16) ----
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    lg = tl.load(LG_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    lb = tl.load(LB_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * lg + lb
    y = y.to(tl.bfloat16)

    # ---- final softmax (bf16 input, fp32 accumulation) ----
    yf = tl.where(mask, y.to(tl.float32), float('-inf'))
    m2 = tl.max(yf, axis=0)
    e2 = tl.exp(yf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(OUT_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x @ self.W0
            x = torch.softmax(x, dim=-1)
            x = x + self.b2
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return torch.softmax(x, dim=-1)

        h = torch.matmul(x, self.W0)  # (M, 2048) via cuBLAS tensor cores
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_row_kernel[(Mrows,)](
            h, self.b2, self.rms3_w, self.ln4_g, self.ln4_b, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
