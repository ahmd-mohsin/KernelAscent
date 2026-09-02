import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 522
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X, W_RMS, G, B, Y,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + EPS_RMS)
    xn = (x * rinv).to(tl.float16)

    # multiply by rms weight in fp16, then scale in fp16 (match reference dtype flow)
    w = tl.load(W_RMS + cols, mask=mask, other=0.0).to(tl.float16)
    h = xn * w
    h = h * SCALE  # fp16 scalar mul

    # LayerNorm: stats & affine in fp32
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    mean = tl.sum(hf, axis=0) / N
    diff = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (hf - mean) * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        scale_h = torch.tensor(1.3348, dtype=torch.float16).item()
        _fused_rms_ln_kernel[(Mrows,)](
            x2d, self.rms0_w, self.ln2_g, self.ln2_b, y,
            N, x2d.stride(0), y.stride(0),
            SCALE=scale_h,
            EPS_RMS=1e-6, EPS_LN=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
