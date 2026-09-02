import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 901
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    G0, B0, W1, G2, B2, G3, B3, G4, B4,
    N, stride_x, stride_y,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    Nf = N.to(tl.float32)

    # ---- LN0 ----
    g = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / Nf
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / Nf
    x = d * tl.math.rsqrt(var + LN_EPS) * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- RMS1 ----
    w = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / Nf
    x = (x * tl.math.rsqrt(ms + RMS_EPS)).to(tl.float16).to(tl.float32) * w
    x = x.to(tl.float16).to(tl.float32)

    # ---- LN2 ----
    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / Nf
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / Nf
    x = d * tl.math.rsqrt(var + LN_EPS) * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- LN3 ----
    g = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / Nf
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / Nf
    x = d * tl.math.rsqrt(var + LN_EPS) * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- LN4 ----
    g = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / Nf
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / Nf
    x = d * tl.math.rsqrt(var + LN_EPS) * g + b

    tl.store(Y + row * stride_y + cols, x.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(rows,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.rms1_w,
            self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            N, x2.stride(0), y.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
