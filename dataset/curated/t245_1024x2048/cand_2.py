import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 245
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_ln_bias_rms_kernel(
    X, OUT, LN_G, LN_B, B1, RMS_W,
    stride_xm, stride_om,
    N, EPS_LN, EPS_RMS, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch's fp32 accumulation for bf16)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    # round to bf16 (layer_norm output dtype)
    y = y.to(tl.bfloat16)

    # + b1 (fp32 opmath, bf16 result)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) + b1).to(tl.bfloat16)

    # RMSNorm: xf = y.float(); xf * rsqrt(mean(xf^2)+eps) -> bf16; * rms_w -> bf16
    xf = y.to(tl.float32)
    xf_m = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf_m * xf_m, axis=0) / N
    rr = 1.0 / tl.sqrt(ms + EPS_RMS)
    t = (xf * rr).to(tl.bfloat16)

    w = tl.load(RMS_W + cols, mask=mask, other=0.0).to(tl.float32)
    t = (t.to(tl.float32) * w).to(tl.bfloat16)

    # * 1.4116 (fp32 opmath, bf16 result)
    out = (t.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_bias_rms_kernel[(Mrows,)](
            x2, out, self.ln0_g, self.ln0_b, self.b1, self.rms2_w,
            x2.stride(0), out.stride(0),
            N, 1e-5, 1e-6, 1.4116,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
