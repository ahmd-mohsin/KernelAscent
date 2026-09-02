import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 865
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_ln_rms_kernel(
    X, OUT, G, B, W,
    N,
    stride_x, stride_o,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm statistics in fp32
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    # cast to fp16 (layer_norm output dtype)
    y16 = y.to(tl.float16)
    # x = x * 1.4994 in fp16
    y16 = (y16 * tl.full((), S1, dtype=tl.float16)).to(tl.float16)

    # RMSNorm in fp32
    yf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)
    z = yf * rrms
    z16 = z.to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    z16 = (z16 * w).to(tl.float16)
    z16 = (z16 * tl.full((), S2, dtype=tl.float16)).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, z16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_rms_kernel[(Mrows,)](
            x, out, self.ln1_g, self.ln1_b, self.rms3_w,
            N,
            x.stride(0), out.stride(0),
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            S1=1.4994,
            S2=1.1216,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
