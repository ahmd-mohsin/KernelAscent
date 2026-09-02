import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 507
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_ln_gelu_softmax(
    X, OUT, G, B, B3,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 internal, biased variance)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    # cast to fp16 (LN output dtype) then back for next op's fp32 opmath
    y = y.to(tl.float16).to(tl.float32)

    # scale
    y = (y * 1.2202).to(tl.float16).to(tl.float32)

    # bias add
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.float16).to(tl.float32)

    # exact GELU (erf-based), fp32 internal, cast to fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    # softmax (fp32 internal)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_gelu_softmax[(m,)](
            x, out, self.ln1_g, self.ln1_b, self.b3,
            n, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return out
