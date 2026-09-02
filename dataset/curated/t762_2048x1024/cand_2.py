import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 762
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_ln_kernel(
    X, B2, B3, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 like PyTorch's opmath, then cast to fp16
    inv_sqrt2 = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    g16 = g.to(tl.float16)

    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)

    # fp16 additions to match reference rounding
    h16 = (g16 + b2) + b3
    h = h16.to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    diff = tl.where(mask, h - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (h - mean) * rstd * gamma + beta
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = torch.matmul(x, self.W0)

        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_gelu_bias_ln_kernel[(m,)](
            h, self.b2, self.b3, self.ln4_g, self.ln4_b, y,
            h.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK, EPS=1e-5,
            num_warps=num_warps,
        )
        return y
