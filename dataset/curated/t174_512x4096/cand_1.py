import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 174
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_ln_relu_add_ln(
    X, OUT, G1, B1, B3, G4, B4,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 math, like PyTorch's bf16 layer_norm)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + b1

    # round to bf16 (matches intermediate storage in reference)
    y = y.to(tl.bfloat16)

    # relu (exact in bf16)
    y = tl.maximum(y, tl.zeros_like(y))

    # + b3 in bf16 (matches bf16 + bf16 in reference)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    y = y + b3

    # LayerNorm 2 (fp32 math)
    z = y.to(tl.float32)
    mean2 = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g4 + b4

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_relu_add_ln[(m,)](
            h, out,
            self.ln1_g, self.ln1_b, self.b3, self.ln4_g, self.ln4_b,
            h.stride(0), out.stride(0),
            N=n, EPS=1e-5, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
