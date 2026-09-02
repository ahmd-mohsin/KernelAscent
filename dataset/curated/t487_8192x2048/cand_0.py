import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 487
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_rms_rms_gelu(X, W1, W2, Y, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    ptr = X + row * stride
    x = tl.load(ptr + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    x1 = xn * w1  # fp16 multiply to match reference

    # RMSNorm 2
    x1f = x1.to(tl.float32)
    ms2 = tl.sum(x1f * x1f, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    xn2 = (x1f * inv2).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    x2 = xn2 * w2  # fp16

    # GELU (erf-based, computed in fp32 like PyTorch's opmath for half)
    xg = x2.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * xg * (1.0 + tl.math.erf(xg * INV_SQRT2))
    tl.store(Y + row * stride + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_rms_gelu[(Mrows,)](
            x, self.rms1_w, self.rms2_w, y, N, x.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return y
