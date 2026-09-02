import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 561
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, eps, scale, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (xn * w).to(tl.float32)
    # exact GELU: 0.5 * h * (1 + erf(h / sqrt(2)))
    g = 0.5 * h * (1.0 + tl.math.erf(h * 0.7071067811865476))
    out = (g * scale)
    tl.store(Y + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, y, N, 1e-6, 1.3603,
            BLOCK_N=BLOCK_N, num_warps=8,
        )
        return y
