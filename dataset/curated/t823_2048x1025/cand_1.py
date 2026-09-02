import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 823
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_scale_ln_ln_scale(
    X, OUT, G2, B2, G3, B3,
    N, stride_x, stride_o,
    S1, S2, EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    x32 = x16.to(tl.float32)

    # x = x * 1.3509 (fp32 math, rounded to fp16 as in reference)
    xs16 = (x32 * S1).to(tl.float16)
    xs = xs16.to(tl.float32)

    # LayerNorm 1 (fp32 internals)
    n = N.to(tl.float32)
    mean1 = tl.sum(xs, axis=0) / n
    d1 = tl.where(mask, xs - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y32 = d1 * rstd1 * g2 + b2
    # cast through fp16 (reference materializes fp16 between the LNs)
    y = y32.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / n
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)

    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z32 = d2 * rstd2 * g3 + b3
    z16 = z32.to(tl.float16)

    # final scale in fp32, round to fp16
    out = (z16.to(tl.float32) * S2).to(tl.float16)
    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_ln_ln_scale[(Mrows,)](
            h, out,
            self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            N, h.stride(0), out.stride(0),
            1.3509, 1.4332, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
