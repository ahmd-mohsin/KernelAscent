import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 633
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_norms_gelu_kernel(
    X_ptr, OUT_ptr,
    W1_ptr, W2_ptr, G_ptr, B_ptr,
    N, stride_x, stride_o,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm 1 (compute in fp32, cast to fp16, multiply by fp16 weight)
    ms1 = tl.sum(xf * xf, axis=0) / N
    inv1 = 1.0 / tl.sqrt(ms1 + RMS_EPS)
    y1 = (xf * inv1).to(tl.float16)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)
    y1 = y1 * w1  # fp16 multiply

    # RMSNorm 2
    xf2 = y1.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, xf2 * xf2, 0.0), axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + RMS_EPS)
    y2 = (xf2 * inv2).to(tl.float16)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0)
    y2 = y2 * w2  # fp16 multiply

    # LayerNorm (fp32 internal math, like PyTorch mixed-precision layer_norm)
    xf3 = y2.to(tl.float32)
    xf3 = tl.where(mask, xf3, 0.0)
    mean = tl.sum(xf3, axis=0) / N
    diff = tl.where(mask, xf3 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ln = diff * rstd * g + b

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * ln * (1.0 + tl.math.erf(ln * INV_SQRT2))

    tl.store(OUT_ptr + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.matmul(x, self.W0)
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_gelu_kernel[(Mrows,)](
            x, out,
            self.rms1_w, self.rms2_w, self.ln3_g, self.ln3_b,
            N, x.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
