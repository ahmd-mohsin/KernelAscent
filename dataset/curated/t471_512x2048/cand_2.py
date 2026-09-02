import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 471
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, LN2G, LN2B, LN4G, LN4B, OUT,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), then round to bf16 like PyTorch op boundary
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (eps=1e-5), fp32 accumulation
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    w2 = tl.load(LN2G + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(LN2B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * w2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax, fp32
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, p, 0.0), axis=0) / N
    d2 = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    w4 = tl.load(LN4G + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(LN4B + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * w4 + b4

    tl.store(OUT + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS bf16 GEMM (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_kernel[(m,)](
            h, self.ln2_g, self.ln2_b, self.ln4_g, self.ln4_b, out,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
