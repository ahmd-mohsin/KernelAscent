import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 587
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X, G, B, W2, W5, Out,
    N: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    x = tl.load(X + base + offs).to(tl.float32)

    # ---- LayerNorm (fp32 math, output cast to bf16) ----
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    y = (d * inv * g + b).to(tl.bfloat16)

    # ---- RMSNorm #1 ----
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    y = (yf * r).to(tl.bfloat16)
    w2 = tl.load(W2 + offs).to(tl.float32)
    y = (y.to(tl.float32) * w2).to(tl.bfloat16)

    # ---- GELU (exact, erf) applied twice, rounding to bf16 each time ----
    SQRT1_2: tl.constexpr = 0.7071067811865476
    yf = y.to(tl.float32)
    yf = yf * 0.5 * (1.0 + tl.math.erf(yf * SQRT1_2))
    y = yf.to(tl.bfloat16)

    yf = y.to(tl.float32)
    yf = yf * 0.5 * (1.0 + tl.math.erf(yf * SQRT1_2))
    y = yf.to(tl.bfloat16)

    # ---- RMSNorm #2 ----
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    y = (yf * r).to(tl.bfloat16)
    w5 = tl.load(W5 + offs).to(tl.float32)
    y = (y.to(tl.float32) * w5).to(tl.bfloat16)

    tl.store(Out + base + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (same op as reference)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)

        _fused_post_kernel[(m,)](
            y, self.ln1_g, self.ln1_b, self.rms2_w, self.rms5_w, out,
            N=n,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
