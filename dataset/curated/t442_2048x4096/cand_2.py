import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 442
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, Y,
    RMS1_W, LN2_G, LN2_B, RMS3_W, LN4_G, LN4_B,
    N, stride_x, stride_y,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_bf = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x_bf.to(tl.float32)

    # ---- RMSNorm 1 (compute in fp32, round to bf16, multiply by bf16 weight) ----
    ms = tl.sum(xf * xf, axis=0) / N
    t = xf * (1.0 / tl.sqrt(ms + RMS_EPS))
    t_bf = t.to(tl.bfloat16)
    w1 = tl.load(RMS1_W + cols, mask=mask, other=0.0)
    x_bf = t_bf * w1  # bf16 arithmetic

    # ---- LayerNorm 2 (fp32 internal math) ----
    xf = x_bf.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g2 = tl.load(LN2_G + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(LN2_B + cols, mask=mask, other=0.0).to(tl.float32)
    yf = d * rstd * g2 + b2
    x_bf = yf.to(tl.bfloat16)

    # ---- RMSNorm 3 ----
    xf = x_bf.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    t = xf * (1.0 / tl.sqrt(ms + RMS_EPS))
    t_bf = t.to(tl.bfloat16)
    w3 = tl.load(RMS3_W + cols, mask=mask, other=0.0)
    x_bf = t_bf * w3

    # ---- LayerNorm 4 ----
    xf = x_bf.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g4 = tl.load(LN4_G + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(LN4_B + cols, mask=mask, other=0.0).to(tl.float32)
    yf = d * rstd * g4 + b4
    out = yf.to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _fused_norms_kernel[(m,)](
            x, y,
            self.rms1_w, self.ln2_g, self.ln2_b,
            self.rms3_w, self.ln4_g, self.ln4_b,
            n, x.stride(0), y.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y @ self.W5
