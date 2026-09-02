import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 723
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_ln_ln_gelu_gelu(
    X, G1, B1, G2, B2, Y,
    stride,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X + row * stride + cols
    x = tl.load(ptr).to(tl.float32)

    # LayerNorm 1 (fp32 math, like PyTorch's half LN)
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(G1 + cols).to(tl.float32)
    b1 = tl.load(B1 + cols).to(tl.float32)
    y = xc * rstd * g1 + b1
    # match intermediate fp16 round-trip of reference
    y = y.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(y, axis=0) / N
    yc = y - mean2
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(G2 + cols).to(tl.float32)
    b2 = tl.load(B2 + cols).to(tl.float32)
    z = yc * rstd2 * g2 + b2
    z = z.to(tl.float16).to(tl.float32)

    # GELU (exact, erf-based) x2 with fp16 round-trip between them
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    z = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))
    z = z.to(tl.float16).to(tl.float32)
    z = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(Y + row * stride + cols, z.to(tl.float16))


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
        # Matmul via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_ln_ln_gelu_gelu[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, out,
            h.stride(0),
            N=N,
            eps=1e-5,
            BLOCK=1024,
            num_warps=8,
        )
        return out
