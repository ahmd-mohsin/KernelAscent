import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 898
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_softmax_gelu_ln(X, G, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, round to fp16 like torch) ----
    mx = tl.max(x, 0)
    e = tl.math.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # relu(relu(p)) is identity since softmax output >= 0

    # ---- exact GELU (erf), fp32 compute, round to fp16 ----
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # ---- LayerNorm (fp32 stats, biased variance) ----
    gm = tl.where(mask, g, 0.0)
    mean = tl.sum(gm, 0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    w = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * w + b

    tl.store(Y + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_gelu_ln[(m,)](
            h, self.ln5_g, self.ln5_b, y,
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
