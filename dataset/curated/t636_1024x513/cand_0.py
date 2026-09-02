import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 636
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_post_gemm(X, B2, RW, LG, LB, OUT,
                     N, stride_x, stride_o,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 compute, round to bf16 like torch) ----
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- add bias (bf16 elementwise -> fp32 opmath, round to bf16) ----
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32 stats, round to bf16, then bf16 scale) ----
    ms = tl.sum(x * x, 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    rw = tl.load(RW + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * rw).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 stats, affine, round to bf16) ----
    mean = tl.sum(tl.where(mask, x, 0.0), 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    lg = tl.load(LG + offs, mask=mask, other=0.0).to(tl.float32)
    lb = tl.load(LB + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * inv * lg + lb).to(tl.bfloat16).to(tl.float32)

    # ---- softmax 2 ----
    x = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x, 0)
    e2 = tl.exp(x - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    y = e2 / s2

    tl.store(OUT + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


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
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        if not h.is_cuda:
            # CPU fallback: reference path
            h = torch.softmax(h, dim=-1)
            h = h + self.b2
            _hf = h.float()
            h = (_hf * torch.rsqrt(_hf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms3_w
            h = F.layer_norm(h, (h.shape[-1],), self.ln4_g, self.ln4_b)
            return torch.softmax(h, dim=-1)

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_gemm[(Mrows,)](
            h, self.b2, self.rms3_w, self.ln4_g, self.ln4_b, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
