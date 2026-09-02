import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 813
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_norms_softmax(
    X, W1, B2, G3, B3, W4, Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x_bf = tl.load(X + row * N + offs, mask=mask, other=0.0)
    xf = x_bf.to(tl.float32)

    # RMSNorm 1 (float math, cast to bf16, bf16 multiply by weight)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    x_bf = (xf * inv).to(tl.bfloat16)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    x_bf = x_bf * w1

    # add bias (bf16)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    x_bf = x_bf + b2

    # LayerNorm (float accumulation, affine in float, cast to bf16)
    xf = x_bf.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    var = tl.sum(xf * xf, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    y_bf = ((xf - mean) * rstd * g3 + b3).to(tl.bfloat16)

    # RMSNorm 2
    yf = y_bf.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    y_bf = (yf * inv2).to(tl.bfloat16)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0)
    y_bf = y_bf * w4

    # Softmax (float accumulation)
    zf = y_bf.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)
    tl.store(Out + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)
        x = x @ self.W0  # cuBLAS GEMM
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_softmax[(Mrows,)](
            x, self.rms1_w, self.b2, self.ln3_g, self.ln3_b, self.rms4_w, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out

    def _forward_ref(self, x):
        x = x @ self.W0
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        x = x + self.b2
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        x = torch.softmax(x, dim=-1)
        return x
