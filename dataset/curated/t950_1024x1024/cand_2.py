import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 950
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_double_rms_gelu(
    X, W1, W2, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(x * x, axis=0) / N
    xn = x * tl.math.rsqrt(ms + EPS)
    xb = xn.to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (xb * w1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ms2 = tl.sum(x * x, axis=0) / N
    xn2 = x * tl.math.rsqrt(ms2 + EPS)
    xb2 = xn2.to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = (xb2 * w2).to(tl.bfloat16).to(tl.float32)

    # GELU (erf, exact)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = x2 * 0.5 * (1.0 + tl.math.erf(x2 * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_double_rms_gelu[(Mrows,)](
            x, self.rms1_w, self.rms2_w, y,
            N, x.stride(0), y.stride(0),
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
