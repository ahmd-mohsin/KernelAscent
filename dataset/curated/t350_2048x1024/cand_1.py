import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 350
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # x = x * 1.4563 (done in bf16, matching PyTorch elementwise semantics:
    # opmath float then round to bf16)
    x = (x.to(tl.float32) * 1.4563).to(tl.bfloat16)

    # LayerNorm in float32 (PyTorch internal accumulation)
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b
    # layer_norm output materialized in bf16
    y = y.to(tl.bfloat16)

    # GELU (exact erf), computed in float, rounded to bf16
    yf = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    gel = gel.to(tl.bfloat16)

    # RMSNorm: _xf = x.float(); xn = (_xf * rsqrt(mean(_xf^2)+1e-6)).to(bf16) * w
    gf = gel.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (gf * rrms).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(M_,)](
            x, self.ln1_g, self.ln1_b, self.rms3_w, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
