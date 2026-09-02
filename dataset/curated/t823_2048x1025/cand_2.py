import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 823
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_double_ln_kernel(
    X, OUT, G2, B2, G3, B3,
    stride_x, stride_o,
    N: tl.constexpr,
    S1: tl.constexpr, S2: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * S1

    # first layernorm
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g2 + b2
    y1 = tl.where(mask, y1, 0.0)

    # second layernorm
    mean2 = tl.sum(y1, axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = (d2 * rstd2 * g3 + b3) * S2

    tl.store(OUT + row * stride_o + cols, y2.to(OUT.dtype.element_ty), mask=mask)


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
        h = torch.matmul(x, self.W0)  # (M, 2048), tensor-core fp16 GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_double_ln_kernel[(Mrows,)](
            h, out,
            self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            h.stride(0), out.stride(0),
            N, 1.3509, 1.4332, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
