import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 498
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _rms_ln_kernel(
    X, W_RMS, G, B, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, then fp16 multiply by weight)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + RMS_EPS)
    y_h = (xf * inv).to(tl.float16)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    y2 = y_h * w  # fp16 multiply, matching reference

    # LayerNorm in fp32 (matches ATen opmath behavior for half inputs)
    y2f = y2.to(tl.float32)
    mean = tl.sum(y2f, axis=0) / N
    diff = tl.where(mask, y2f - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y2f - mean) * rstd * g + b

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        x = x @ self.W0
        x = x @ self.W1
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_ln_kernel[(Mrows,)](
            x, self.rms2_w, self.ln3_g, self.ln3_b, out,
            x.stride(0), out.stride(0),
            N=N, RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
