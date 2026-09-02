import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 724
M, D, DT = 2048, 512, torch.bfloat16

INV_SQRT2 = 0.7071067811865476


@triton.jit
def _gelu2_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # First GELU (exact, erf-based), round to bf16 like PyTorch does
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # Second GELU, round to bf16
    g2 = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 (matches _xf = x.float(); mean of squares)
    ms = tl.sum(tl.where(mask, g2 * g2, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)

    y = (g2 * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores (bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _gelu2_rmsnorm_kernel[(Mrows,)](
            h, self.rms3_w, y,
            N=N, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
