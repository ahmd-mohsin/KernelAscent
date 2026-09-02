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
    X, G, B, W2, W5, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = row * N + offs

    x = tl.load(X + ptr).to(tl.float32)

    # ---- LayerNorm (fp32 math, eps=1e-5), output rounded to bf16 ----
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    xhat = d * tl.rsqrt(var + 1e-5)
    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    x = (xhat * g + b).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm #1 (fp32 math on bf16-rounded input, eps=1e-6) ----
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + offs).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact, erf-based, fp32 math) x2, rounding to bf16 between ----
    SQRT1_2: tl.constexpr = 0.7071067811865476
    x = (x * 0.5 * (1.0 + tl.math.erf(x * SQRT1_2))).to(tl.bfloat16).to(tl.float32)
    x = (x * 0.5 * (1.0 + tl.math.erf(x * SQRT1_2))).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm #2 (eps=1e-6) ----
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w5 = tl.load(W5 + offs).to(tl.float32)
    y = (x * w5).to(tl.bfloat16)

    tl.store(Y + ptr, y)


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
        # GEMM via cuBLAS (tensor cores, fp32 accumulate)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_post_kernel[(m,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, self.rms5_w, out,
            N=n, BLOCK=n,
            num_warps=8,
        )
        return out
