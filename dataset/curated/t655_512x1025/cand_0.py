import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 655
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_scale_ln_rms_kernel(
    X, Y, G, B, W,
    stride_xm, stride_ym,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * SCALE

    # LayerNorm (fp32 accumulation, biased variance, eps=1e-5)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = xc * rstd * g + b

    # cast to fp16 (matches F.layer_norm output dtype), then back to fp32 for RMS
    ln_h = ln.to(tl.float16)
    xf = ln_h.to(tl.float32)

    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)

    y_h = (xf * rrms).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = y_h * w  # fp16 multiply, matching PyTorch semantics

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        grid = (m,)
        _fused_scale_ln_rms_kernel[grid](
            h, y, self.ln2_g, self.ln2_b, self.rms3_w,
            h.stride(0), y.stride(0),
            N=n,
            SCALE=1.4538,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
