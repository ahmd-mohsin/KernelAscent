import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 574
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_post_matmul(
    X, LNG, LNB, R3, R5, OUT,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    x = tl.load(X + base + offs).to(tl.float32)

    # ---- exact GELU (erf based), computed in fp32, rounded to bf16 like PyTorch ----
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (eps=1e-5), fp32 accumulation, bf16 output ----
    mean = tl.sum(g, axis=0) / N
    diff = g - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    ln_g = tl.load(LNG + offs).to(tl.float32)
    ln_b = tl.load(LNB + offs).to(tl.float32)
    ln = diff * inv_std * ln_g + ln_b
    ln = ln.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm #1 (eps=1e-6): normalize in fp32, cast bf16, then * weight ----
    ms = tl.sum(ln * ln, axis=0) / N
    r = ln * (1.0 / tl.sqrt(ms + 1e-6))
    r = r.to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(R3 + offs).to(tl.float32)
    r = r * w3
    r = r.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 math, bf16 output) ----
    mx = tl.max(r, axis=0)
    e = tl.exp(r - mx)
    s = tl.sum(e, axis=0)
    sm = e / s
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm #2 (eps=1e-6) ----
    ms2 = tl.sum(sm * sm, axis=0) / N
    r2 = sm * (1.0 / tl.sqrt(ms2 + 1e-6))
    r2 = r2.to(tl.bfloat16).to(tl.float32)
    w5 = tl.load(R5 + offs).to(tl.float32)
    out = r2 * w5

    tl.store(OUT + base + offs, out.to(tl.bfloat16))


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
        # GEMM via cuBLAS (tensor cores)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        _fused_post_matmul[(rows,)](
            y, self.ln2_g, self.ln2_b, self.rms3_w, self.rms5_w, out,
            N=N, BLOCK=N,
            num_warps=16,
        )
        return out
