import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 508
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_gelu_ln_bias(
    X, OUT, G, B, B4,
    stride_xm, stride_om,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, round to fp16 as intermediate tensor would be)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom
    s = s.to(tl.float16).to(tl.float32)

    # gelu (erf-based, fp32 opmath, round to fp16)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = s * 0.5 * (1.0 + tl.math.erf(s * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # layernorm (fp32 stats, round to fp16)
    g_masked = tl.where(mask, g, 0.0)
    mean = tl.sum(g_masked, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g - mean) * rstd * w + bb
    y = y.to(tl.float16).to(tl.float32)

    # bias add (fp32 opmath), final fp16
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y + b4).to(tl.float16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_gelu_ln_bias[(m,)](
            h, out, self.ln3_g, self.ln3_b, self.b4,
            h.stride(0), out.stride(0),
            N=n, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
