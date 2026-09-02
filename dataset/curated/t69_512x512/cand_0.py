import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 69
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_rms_softmax_gelu(X, W, Out, N,
                                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # ---- GELU (exact, computed in fp32 like PyTorch's half opmath) ----
    x = tl.load(X + base + offs).to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    gh = g.to(tl.float16)          # round to fp16 (matches PyTorch intermediate)

    # ---- RMSNorm (stats in fp32, applied, cast to fp16, scaled by w in fp16) ----
    gf = gh.to(tl.float32)
    ms = tl.sum(gf * gf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    normed = (gf * r).to(tl.float16)
    w = tl.load(W + offs)          # fp16 weight
    y = normed * w                 # fp16 multiply (matches half*half)

    # ---- Softmax (fp32 accumulation, fp16 output, like PyTorch half softmax) ----
    yf = y.to(tl.float32)
    mx = tl.max(yf, axis=0)
    e = tl.math.exp(yf - mx)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # ---- Final GELU ----
    sf = sm.to(tl.float32)
    out = 0.5 * sf * (1.0 + tl.math.erf(sf * INV_SQRT2))
    tl.store(Out + base + offs, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        _fused_gelu_rms_softmax_gelu[(rows,)](
            h, self.rms2_w, out, N,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
