import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 37
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_gelu_relu_ln_scale(
    X, G, B, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32, rounded to fp16 like PyTorch storage
    inv_sqrt2 = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    g = g.to(tl.float16).to(tl.float32)

    # ReLU
    r = tl.maximum(g, 0.0)
    r = tl.where(mask, r, 0.0)

    # LayerNorm (stats in fp32)
    mean = tl.sum(r, axis=0) / N
    diff = tl.where(mask, r - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (r - mean) * rstd * gamma + beta
    y = y.to(tl.float16).to(tl.float32)  # match LN's fp16 output rounding

    y = y * SCALE

    tl.store(OUT + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # tensor-core GEMM
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_relu_ln_scale[(m,)](
            y, self.ln3_g, self.ln3_b, out,
            y.stride(0), out.stride(0),
            N=n, SCALE=1.4798, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
