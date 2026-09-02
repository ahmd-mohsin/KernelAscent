import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 394
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_norm_kernel(
    Y, OUT, W1, G2, B2, W4, W5,
    stride,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * stride + offs

    # ---- input row (fp16 from matmul) ----
    x16 = tl.load(Y + base)

    # ---- RMSNorm 1 ----
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, 0) / N
    x16 = (xf * tl.math.rsqrt(ms + 1e-6)).to(tl.float16)
    x16 = x16 * tl.load(W1 + offs)  # fp16 multiply (matches PyTorch)

    # ---- LayerNorm (compute in fp32, cast to fp16) ----
    xf = x16.to(tl.float32)
    mean = tl.sum(xf, 0) / N
    d = xf - mean
    var = tl.sum(d * d, 0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G2 + offs).to(tl.float32)
    b = tl.load(B2 + offs).to(tl.float32)
    x16 = (d * rstd * g + b).to(tl.float16)

    # ---- GELU (exact erf, fp32 math, cast to fp16) ----
    xf = x16.to(tl.float32)
    xg = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x16 = xg.to(tl.float16)

    # ---- RMSNorm 4 ----
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, 0) / N
    x16 = (xf * tl.math.rsqrt(ms + 1e-6)).to(tl.float16)
    x16 = x16 * tl.load(W4 + offs)

    # ---- RMSNorm 5 ----
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, 0) / N
    x16 = (xf * tl.math.rsqrt(ms + 1e-6)).to(tl.float16)
    x16 = x16 * tl.load(W5 + offs)

    tl.store(OUT + base, x16)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # tensor-core GEMM
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        _fused_norm_kernel[(rows,)](
            y, out,
            self.rms1_w, self.ln2_g, self.ln2_b, self.rms4_w, self.rms5_w,
            y.stride(0),
            N=N, BLOCK=N,
            num_warps=8,
        )
        return out
