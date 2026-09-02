import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 392
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_norms_kernel(
    X_ptr, Y_ptr,
    W1_ptr, W2_ptr, G_ptr, B_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    base = row * N + cols

    # ---- load row (fp16) ----
    x16 = tl.load(X_ptr + base)

    # ---- RMSNorm 1 (compute in fp32, cast to fp16, mul by fp16 weight) ----
    xf = x16.to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    x16 = (xf * r1).to(tl.float16)
    w1 = tl.load(W1_ptr + cols)
    x16 = x16 * w1  # fp16 multiply (correctly rounded, matches torch)

    # ---- RMSNorm 2 ----
    xf = x16.to(tl.float32)
    ms2 = tl.sum(xf * xf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    x16 = (xf * r2).to(tl.float16)
    w2 = tl.load(W2_ptr + cols)
    x16 = x16 * w2

    # ---- Softmax (fp32 accumulation like torch, cast back to fp16) ----
    xf = x16.to(tl.float32)
    mmax = tl.max(xf, axis=0)
    e = tl.exp(xf - mmax)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)

    # ---- LayerNorm (fp32 internal, like torch) ----
    pf = p16.to(tl.float32)
    mu = tl.sum(pf, axis=0) / N
    d = pf - mu
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + cols).to(tl.float32)
    b = tl.load(B_ptr + cols).to(tl.float32)
    y16 = (d * inv * g + b).to(tl.float16)

    # ---- final scale: half tensor * python float -> fp32 math -> half ----
    out = (y16.to(tl.float32) * 1.0194).to(tl.float16)

    tl.store(Y_ptr + base, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM with fp32 accumulate (same as reference)
        x = x @ self.W0

        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x * 1.0194

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _fused_norms_kernel[(m,)](
            x, y,
            self.rms1_w, self.rms2_w, self.ln4_g, self.ln4_b,
            N=n, BLOCK=n,
            num_warps=8,
        )
        return y
