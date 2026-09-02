import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 589
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_ln_bias_relu_gelu2(
    Y_ptr, G_ptr, B_ptr, B2_ptr, OUT_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offs = row * N + cols

    # Load matmul output, compute LayerNorm stats in fp32 (matches PyTorch bf16 path)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(y, axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm affine in fp32, round to bf16 (as PyTorch does between ops)
    x = (y - mean) * rstd * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # Add bias in fp32 opmath, round to bf16
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    # ReLU (exact in any precision)
    x = tl.maximum(x, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # First exact GELU (erf), fp32 opmath, round to bf16
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    # Second exact GELU, store as bf16
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    tl.store(OUT_ptr + offs, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 tensor-core matmul
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_bias_relu_gelu2[(Mrows,)](
            y, self.ln1_g, self.ln1_b, self.b2, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
