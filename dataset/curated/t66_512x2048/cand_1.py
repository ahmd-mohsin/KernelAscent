import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 66
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, W1, W3, G4, B4, OUT,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1 (fp32 compute, cast to fp16, then fp16 mul by weight)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (x * r).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    xh = xh * w1

    # GELU (exact, erf-based; opmath fp32 as PyTorch does for half)
    xf = xh.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    xh = g.to(tl.float16)

    # RMSNorm 2
    xf = xh.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * r).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    xh = xh * w3

    # LayerNorm (fp32 compute)
    xf = xh.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * inv * g4 + b4).to(tl.float16)

    # final scale (fp16 mul, matching reference)
    y = y * SCALE

    tl.store(OUT + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_kernel[(Mrows,)](
            x, self.rms1_w, self.rms3_w, self.ln4_g, self.ln4_b, out,
            N, x.stride(0), out.stride(0),
            SCALE=1.2037,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
