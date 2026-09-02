import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 943
M, D, DT = 8192, 1024, torch.float16

@triton.jit
def _fused_rms_relu_gelu(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # RMSNorm
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.float16).to(tl.float32)  # round to fp16 like reference
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    xw = (xn * w).to(tl.float16).to(tl.float32)   # fp16 rounding of product
    # ReLU (exact in fp16 domain)
    xr = tl.maximum(xw, 0.0)
    # GELU (erf-based, computed in fp32 like PyTorch opmath)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = xr * 0.5 * (1.0 + tl.math.erf(xr * INV_SQRT2))
    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_relu_gelu[(Mrows,)](
            x, self.rms0_w, y, N, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
