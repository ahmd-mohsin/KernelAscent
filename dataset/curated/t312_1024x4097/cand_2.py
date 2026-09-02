import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 312
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_ln_scale_rms_kernel(
    X, G, B, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, output rounded to fp16 like F.layer_norm on half)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd) * g + b
    y = y.to(tl.float16)  # F.layer_norm output dtype

    # x * 1.0028 : half tensor * scalar -> compute in fp32, round to fp16
    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    # RMSNorm in fp32
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    z = (yf * r).to(tl.float16)  # cast back to fp16 as in reference

    # z * rms3_w : half*half computed in fp32, rounded to fp16
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z.to(tl.float32) * w).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_ln_scale_rms_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms3_w, out,
            h.stride(0), out.stride(0),
            N=N,
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            SCALE=1.0028,
            BLOCK=512,
            num_warps=4,
        )
        return out
