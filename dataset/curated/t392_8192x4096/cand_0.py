import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 392
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_norms_softmax_ln_kernel(
    X, W1, W2, G, B, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X + row * stride_x + offs)          # fp16
    w1 = tl.load(W1 + offs)                          # fp16
    w2 = tl.load(W2 + offs)                          # fp16

    # ---- RMSNorm 1 (compute in fp32, cast to fp16, multiply weight in fp16) ----
    xf = x.to(tl.float32)
    r = tl.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    x = (xf * r).to(tl.float16) * w1

    # ---- RMSNorm 2 ----
    xf = x.to(tl.float32)
    r = tl.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    x = (xf * r).to(tl.float16) * w2

    # ---- Softmax (fp32 accumulation, fp16 output) ----
    xf = x.to(tl.float32)
    mmax = tl.max(xf, axis=0)
    e = tl.exp(xf - mmax)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # ---- LayerNorm (fp32 compute, fp16 output) ----
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = xf - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    y = (d * rstd * g + b).to(tl.float16)

    # ---- scale by 1.0194 (opmath fp32, output fp16) ----
    y = (y.to(tl.float32) * 1.0194).to(tl.float16)

    tl.store(Out + row * stride_o + offs, y)


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
        # GEMM via cuBLAS (tensor cores)
        x = x @ self.W0

        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)

        _fused_norms_softmax_ln_kernel[(m,)](
            x, self.rms1_w, self.rms2_w, self.ln4_g, self.ln4_b, out,
            x.stride(0), out.stride(0),
            N=n, BLOCK=n,
            num_warps=8,
        )
        return out
