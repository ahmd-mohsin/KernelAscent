import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 264
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_norms_gelu(
    X, W1, G2, B2, G3, B3, Out,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = row * N + offs

    x = tl.load(X + ptr).to(tl.float32)

    # ---- RMSNorm (computed in fp32, rounded to fp16 like reference) ----
    ms = tl.sum(x * x, axis=0) / N
    y = x * tl.math.rsqrt(ms + 1e-6)
    y = y.to(tl.float16).to(tl.float32)
    w = tl.load(W1 + offs).to(tl.float32)
    y = (y * w).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    m = tl.sum(y, axis=0) / N
    d = y - m
    v = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(v + 1e-5)
    g2 = tl.load(G2 + offs).to(tl.float32)
    b2 = tl.load(B2 + offs).to(tl.float32)
    y = (d * inv * g2 + b2).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 3 ----
    m = tl.sum(y, axis=0) / N
    d = y - m
    v = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(v + 1e-5)
    g3 = tl.load(G3 + offs).to(tl.float32)
    b3 = tl.load(B3 + offs).to(tl.float32)
    y = (d * inv * g3 + b3).to(tl.float16).to(tl.float32)

    # ---- exact GELU ----
    out = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    tl.store(Out + ptr, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        x = torch.matmul(x, self.W0)
        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)
        _fused_norms_gelu[(rows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b, out,
            N=N, BLOCK=N,
            num_warps=4,
        )
        return out
