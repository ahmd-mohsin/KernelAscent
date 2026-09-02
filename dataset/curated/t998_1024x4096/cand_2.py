import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 998
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _double_ln_kernel(
    X, Y, G1, B1, G2, B2,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # First LayerNorm
    mean1 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(xc * xc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = xc * rstd1 * g1 + b1
    # round to bf16 to match intermediate cast in reference
    y1 = y1.to(tl.bfloat16).to(tl.float32)

    # Second LayerNorm
    mean2 = tl.sum(tl.where(mask, y1, 0.0), axis=0) / N
    yc = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = yc * rstd2 * g2 + b2

    tl.store(Y + row * N + cols, y2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _double_ln_kernel[(rows,)](
            h, y,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            N, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
