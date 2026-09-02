import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 805
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_ln_relu_gelu_ln(
    X, Y, G1, B1, G2, B2,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    h = xc * rstd * g1 + b1

    # ReLU
    h = tl.maximum(h, 0.0)

    # GELU (exact, erf-based)
    h = 0.5 * h * (1.0 + tl.math.erf(h * 0.7071067811865476))

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    hc = tl.where(mask, h - mean2, 0.0)
    var2 = tl.sum(hc * hc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = hc * rstd2 * g2 + b2

    tl.store(Y + row * stride_y + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 16 if BLOCK >= 4096 else 8
        _fused_ln_relu_gelu_ln[(Mrows,)](
            h, y, self.ln1_g, self.ln1_b, self.ln4_g, self.ln4_b,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
