import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 826
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_ln_rms_kernel(
    X, Y, G, B, W,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # round to fp16 as layer_norm output does
    y = (xc * rstd * g + b).to(tl.float16)

    # x * 1.1102 in fp32 opmath, rounded to fp16
    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    # RMSNorm in fp32 on the fp16 values
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)

    yr = (yf * rrms).to(tl.float16)  # .to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (yr.to(tl.float32) * w).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_rms_kernel[(rows,)](
            x, y, self.ln0_g, self.ln0_b, self.rms2_w,
            x.stride(0), y.stride(0),
            N=N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            SCALE=1.1102,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
