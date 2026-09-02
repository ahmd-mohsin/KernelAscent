import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 931
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _relu_double_ln_kernel(
    X, OUT, G2, B2, G3, B3,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu (in bf16, same as reference)
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # first layernorm (fp32 math)
    mean1 = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d1 = tl.where(mask, xf - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g2 + b2
    # round to bf16 to match intermediate dtype in reference
    y = y.to(tl.bfloat16).to(tl.float32)

    # second layernorm
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)

    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g3 + b3

    tl.store(OUT + row * stride_o + cols, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _relu_double_ln_kernel[(Mrows,)](
            h, out,
            self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            N, h.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
