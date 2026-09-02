import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 775
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X, G, B, B4, W5, Out,
    N: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, round to bf16) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + EPS_LN)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * inv * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax #1 (fp32 math, round to bf16) ----
    xm = tl.where(mask, x, float("-inf"))
    mx = tl.max(xm, axis=0)
    e = tl.where(mask, tl.exp(x - mx), 0.0)
    x = e / tl.sum(e, axis=0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax #2 ----
    xm = tl.where(mask, x, float("-inf"))
    mx = tl.max(xm, axis=0)
    e = tl.where(mask, tl.exp(x - mx), 0.0)
    x = e / tl.sum(e, axis=0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- Add bias (fp32 opmath, round to bf16) ----
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b4
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32), round to bf16, then multiply by weight ----
    x2 = tl.where(mask, x * x, 0.0)
    rms = tl.math.rsqrt(tl.sum(x2, axis=0) / N + EPS_RMS)
    y = (x * rms).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W5 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(Out + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = x @ self.W0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            x = x + self.b4
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms5_w
            return x

        h = torch.matmul(x, self.W0)  # cuBLAS bf16 GEMM
        h = h.contiguous()
        rows, N = h.shape[0], h.shape[-1]
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.b4, self.rms5_w, out,
            N=N, EPS_LN=1e-5, EPS_RMS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
