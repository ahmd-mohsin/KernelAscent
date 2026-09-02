import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 574
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X, G, B, W3, W5, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    x = tl.load(X + base + offs).to(tl.float32)

    # --- exact GELU (erf), computed in fp32, rounded back to bf16 ---
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # --- LayerNorm (fp32 accumulation, bf16 output) ---
    mean = tl.sum(g, axis=0) / N
    d = g - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    gam = tl.load(G + offs).to(tl.float32)
    bet = tl.load(B + offs).to(tl.float32)
    y = d * rstd * gam + bet
    y = y.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm 3 (fp32 math, bf16 round, then bf16*bf16 weight mul) ---
    ms = tl.sum(y * y, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(W3 + offs).to(tl.float32)
    z = (y * r).to(tl.bfloat16).to(tl.float32)
    z = (z * w3).to(tl.bfloat16).to(tl.float32)

    # --- Softmax (fp32 accumulation, bf16 output) ---
    mx = tl.max(z, axis=0)
    e = tl.exp(z - mx)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm 5 ---
    ms2 = tl.sum(p * p, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w5 = tl.load(W5 + offs).to(tl.float32)
    out = (p * r2).to(tl.bfloat16).to(tl.float32)
    out = (out * w5).to(tl.bfloat16)

    tl.store(Y + base + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        _fused_post_kernel[(rows,)](
            h, self.ln2_g, self.ln2_b, self.rms3_w, self.rms5_w, y,
            N=N, BLOCK=N,
            num_warps=16,
        )
        return y
