import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 503
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_post_kernel(
    X, W1, B2, G3, Bt3, G5, Bt5, Out,
    N, stride_x, stride_o,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm (compute in fp32, cast to fp16, then scale by fp16 weight)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    xn = (xf * r).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    xn = xn * w1

    # add bias (fp16)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    xn = xn + b2

    # LayerNorm 3 (fp32 math, fp16 out)
    xf = xn.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    bt3 = tl.load(Bt3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * inv * g3 + bt3).to(tl.float16)

    # GELU (erf variant, fp32 math, fp16 out)
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    y = g.to(tl.float16)

    # LayerNorm 5
    xf = y.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    g5 = tl.load(G5 + cols, mask=mask, other=0.0).to(tl.float32)
    bt5 = tl.load(Bt5 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (d * inv * g5 + bt5).to(tl.float16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, self.ln3_g, self.ln3_b,
            self.ln5_g, self.ln5_b, out,
            N, x.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK, num_warps=4,
        )
        return out
