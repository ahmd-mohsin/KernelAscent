import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 498
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X, W_RMS, G, B, OUT,
    stride_xm, stride_om,
    N: tl.constexpr,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, then fp16 multiply by weight)
    ms = tl.sum(x * x, axis=0) / N
    r = tl.rsqrt(ms + RMS_EPS)
    y_h = (x * r).to(tl.float16)
    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    y_h = y_h * w  # fp16 multiply, matches reference

    # LayerNorm in fp32
    y = y_h.to(tl.float32)
    mean = tl.sum(y, axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.rsqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = d * rstd * g + b

    tl.store(OUT + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x @ self.W1
        x = x.contiguous()
        M_, N_ = x.shape
        out = torch.empty_like(x)
        _fused_rms_ln_kernel[(M_,)](
            x, self.rms2_w, self.ln3_g, self.ln3_b, out,
            x.stride(0), out.stride(0),
            N=N_,
            RMS_EPS=1e-6,
            LN_EPS=1e-5,
            BLOCK=triton.next_power_of_2(N_),
            num_warps=8,
        )
        return out
