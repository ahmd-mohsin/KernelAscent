import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 723
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _ln_ln_gelu_gelu_kernel(
    X, G1, B1, G2, B2, Y,
    N: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    base = row * N

    x = tl.load(X + base + cols).to(tl.float32)

    # LayerNorm 1 (stats in fp32, output cast to fp16 to match PyTorch)
    mean1 = tl.sum(x, axis=0) / N
    d1 = x - mean1
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + cols).to(tl.float32)
    b1 = tl.load(B1 + cols).to(tl.float32)
    y = d1 * rstd1 * g1 + b1
    y = y.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(y, axis=0) / N
    d2 = y - mean2
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(G2 + cols).to(tl.float32)
    b2 = tl.load(B2 + cols).to(tl.float32)
    z = d2 * rstd2 * g2 + b2
    z = z.to(tl.float16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # GELU (exact, erf-based), cast to fp16 between ops to match reference
    z = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))
    z = z.to(tl.float16).to(tl.float32)
    z = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(Y + base + cols, z.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        _ln_ln_gelu_gelu_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, out,
            N, 1e-5,
            BLOCK=N,
            num_warps=8,
        )
        return out
