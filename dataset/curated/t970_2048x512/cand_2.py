import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 970
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _gelu2_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # first GELU (exact, erf-based), computed in fp32, rounded to fp16 like PyTorch
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)

    # second GELU
    g2 = g1 * 0.5 * (1.0 + tl.math.erf(g1 * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)

    # RMSNorm in fp32 over the fp16-rounded values
    ms = tl.sum(tl.where(mask, g2 * g2, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    normed = (g2 * inv).to(tl.float16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (normed * w).to(tl.float16)

    tl.store(Y_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 TC matmul
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _gelu2_rmsnorm_kernel[(Mrows,)](
            x, self.rms3_w, y,
            N, x.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
