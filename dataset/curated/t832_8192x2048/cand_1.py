import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 832
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _rms_kernel(X, W, Y, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (x * r).to(tl.float16) * w
    tl.store(Y + row * D + offs, y, mask=mask)


@triton.jit
def _rms_scale_ln_kernel(X, W2, G, B, Y, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # RMSNorm (fp32 math, cast to fp16 like reference)
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    y16 = (x * r).to(tl.float16) * w2

    # scalar multiply: PyTorch computes half*scalar with fp32 opmath -> half
    y16 = (y16.to(tl.float32) * 1.1618).to(tl.float16)

    # LayerNorm in fp32 (matches PyTorch's fp32 accumulation for half input)
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / D
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    rstd = tl.math.rsqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    out = d * rstd * g + b
    tl.store(Y + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x = x.contiguous().view(-1, d)
        m = x.shape[0]
        BLOCK = triton.next_power_of_2(d)

        # Fused RMSNorm 0
        t = torch.empty_like(x)
        _rms_kernel[(m,)](x, self.rms0_w, t, d, BLOCK=BLOCK, num_warps=8)

        # GEMM (cuBLAS tensor cores)
        h = t @ self.W1

        # Fused RMSNorm + scale + LayerNorm
        out = torch.empty_like(h)
        _rms_scale_ln_kernel[(m,)](
            h, self.rms2_w, self.ln4_g, self.ln4_b, out, d, BLOCK=BLOCK, num_warps=8
        )

        return out.view(orig_shape)
