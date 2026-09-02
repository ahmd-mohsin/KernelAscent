import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 592
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X, W_RMS, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + RMS_EPS)

    # cast to fp16, multiply by rms weight in fp16 (matches reference)
    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    t16 = (xf * rinv).to(tl.float16) * w

    # LayerNorm: compute in fp32
    tf = t16.to(tl.float32)
    mean = tl.sum(tf, axis=0) / N
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (tf - mean) * rstd * g + b

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = x @ self.W0

        M_, N_ = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _fused_rms_ln_kernel[(M_,)](
            h, self.rms1_w, self.ln2_g, self.ln2_b, y,
            h.stride(0), y.stride(0),
            N=N_,
            RMS_EPS=1e-6,
            LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
