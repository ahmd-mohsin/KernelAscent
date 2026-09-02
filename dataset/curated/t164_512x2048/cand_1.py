import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 164
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, W1, W2, G, B, Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = row * N + cols

    # ---- RMSNorm 1 ----
    x = tl.load(X + ptr).to(tl.float32)
    ms1 = tl.sum(x * x, axis=0) / N
    y = (x * tl.math.rsqrt(ms1 + 1e-6)).to(tl.bfloat16)
    w1 = tl.load(W1 + cols).to(tl.float32)
    y = (y.to(tl.float32) * w1).to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    z = (yf * tl.math.rsqrt(ms2 + 1e-6)).to(tl.bfloat16)
    w2 = tl.load(W2 + cols).to(tl.float32)
    z = (z.to(tl.float32) * w2).to(tl.bfloat16)

    # ---- scale by 1.0772 (bf16 rounding like PyTorch) ----
    z = (z.to(tl.float32) * 1.0772).to(tl.bfloat16)

    # ---- LayerNorm (fp32 accumulation, eps=1e-5) ----
    zf = z.to(tl.float32)
    mean = tl.sum(zf, axis=0) / N
    d = zf - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    o = (d * rstd * g + b).to(tl.bfloat16)

    tl.store(Out + ptr, o)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        _fused_norms_kernel[(m,)](
            x, self.rms1_w, self.rms2_w, self.ln4_g, self.ln4_b, out,
            N=n, BLOCK=n,
            num_warps=8,
        )
        return out @ self.W5
