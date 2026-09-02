import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 770
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_bias_rms_ln_kernel(
    X, B0, W1, G2, B2, OUT,
    stride_x, stride_o,
    D_: tl.constexpr,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # x = x + b0  (bf16 add: compute in fp32, round to bf16)
    xf = x.to(tl.float32) + b0.to(tl.float32)
    x_bf = xf.to(tl.bfloat16)

    # RMSNorm in fp32
    xf2 = x_bf.to(tl.float32)
    ms = tl.sum(xf2 * xf2, axis=0) / D_
    r = tl.math.rsqrt(ms + RMS_EPS)
    y_bf = (xf2 * r).to(tl.bfloat16)

    # multiply by rms1_w in bf16 semantics (fp32 compute, bf16 round)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    y_bf = (y_bf.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm in fp32
    yf = y_bf.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / D_
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    inv = tl.math.rsqrt(var + LN_EPS)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (yf - mean) * inv * g2 + b2

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_bias_rms_ln_kernel[(m,)](
            x2, self.b0, self.rms1_w, self.ln2_g, self.ln2_b, out,
            x2.stride(0), out.stride(0),
            d, 1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
