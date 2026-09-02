import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 361
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_ln_softmax_ln_act(
    Y, OUT, G1, B1, G3, B3,
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1
    mean = tl.sum(y, axis=0) / N
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    h = (y - mean) * rstd * g1 + b1
    h = h.to(tl.bfloat16).to(tl.float32)  # match dtype rounding

    # Softmax
    h_m = tl.where(mask, h, float('-inf'))
    mx = tl.max(h_m, axis=0)
    e = tl.exp(h_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 3
    mean2 = tl.sum(p, axis=0) / N
    d2 = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (p - mean2) * rstd2 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # ReLU
    z = tl.maximum(z, 0.0)
    z = z.to(tl.bfloat16).to(tl.float32)

    # GELU (erf-based, exact)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = z * 0.5 * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(OUT + row * stride + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_softmax_ln_act[(m,)](
            y, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            n, y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
