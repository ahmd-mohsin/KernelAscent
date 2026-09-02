import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 435
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


@triton.jit
def _ln_rms_gelu_kernel(X, G, B, W, Y, N, eps_ln, eps_rms, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, output rounded to fp16 like F.layer_norm on half)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y_h = y.to(tl.float16)          # round to half (layer_norm output dtype)

    # RMSNorm: _xf = x.float(); (_xf * rsqrt(mean(_xf^2)+eps)).half() * w
    xf = y_h.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps_rms)
    z_h = (xf * r).to(tl.float16)   # round to half before weight mul
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z_h = (z_h.to(tl.float32) * w).to(tl.float16)  # half*half mul (fp32 opmath, round)

    # GELU (erf form, computed in fp32 like CUDA half kernel)
    t = z_h.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * t * (1.0 + tl.math.erf(t * INV_SQRT2))
    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        h = x @ self.W0  # cuBLAS fp16 tensor cores, fp32 accumulate

        m, n = h.shape
        s = torch.empty_like(h)
        _softmax_kernel[(m,)](h, s, n, BLOCK=triton.next_power_of_2(n), num_warps=8)

        z = s @ self.W2

        m2, n2 = z.shape
        out = torch.empty_like(z)
        _ln_rms_gelu_kernel[(m2,)](
            z, self.ln3_g, self.ln3_b, self.rms4_w, out,
            n2, 1e-5, 1e-6, BLOCK=triton.next_power_of_2(n2), num_warps=4,
        )
        return out
