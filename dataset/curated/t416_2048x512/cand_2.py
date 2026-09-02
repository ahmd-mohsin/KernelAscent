import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 416
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    LN0G, LN0B, R1W, LN2G, LN2B, R3W,
    N, stride_x, stride_y,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, bf16 output like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.math.rsqrt(var + LN_EPS)
    g0 = tl.load(LN0G + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(LN0B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd * g0 + b0).to(tl.bfloat16)

    # ---- RMSNorm 1 (fp32 stats, cast bf16, weight mul in bf16) ----
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + RMS_EPS)
    y = (yf * r).to(tl.bfloat16)
    w1 = tl.load(R1W + cols, mask=mask, other=0.0)
    y = y * w1  # bf16 * bf16 -> bf16

    # ---- LayerNorm 2 ----
    xf = y.to(tl.float32)
    mean2 = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    xc2 = tl.where(mask, xf - mean2, 0.0)
    var2 = tl.sum(xc2 * xc2, axis=0) / N
    rstd2 = tl.math.rsqrt(var2 + LN_EPS)
    g2 = tl.load(LN2G + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(LN2B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc2 * rstd2 * g2 + b2).to(tl.bfloat16)

    # ---- RMSNorm 3 ----
    yf = y.to(tl.float32)
    ms3 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r3 = tl.math.rsqrt(ms3 + RMS_EPS)
    y = (yf * r3).to(tl.bfloat16)
    w3 = tl.load(R3W + cols, mask=mask, other=0.0)
    y = y * w3  # bf16 * bf16 -> bf16

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            yf = y.float()
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms1_w
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            yf = y.float()
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_norm_kernel[(rows,)](
            x2, out,
            self.ln0_g, self.ln0_b, self.rms1_w,
            self.ln2_g, self.ln2_b, self.rms3_w,
            N, x2.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
