import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 766
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_rms_ln_gelu(
    X, W_RMS, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_bf = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x_bf.to(tl.float32)

    # RMSNorm (computed in fp32, rounded to bf16, then * bf16 weight -> bf16)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xr_bf = (xf * inv).to(tl.bfloat16)
    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    y_bf = xr_bf * w  # bf16 * bf16 -> bf16 (matches reference rounding)

    # LayerNorm on bf16 input, internal fp32 math, output rounded to bf16
    yf = y_bf.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    z_bf = (d * rstd * g + b).to(tl.bfloat16)

    # GELU (exact, erf) on bf16 input with fp32 internal math
    zf = z_bf.to(tl.float32)
    out = 0.5 * zf * (1.0 + tl.math.erf(zf * 0.70710678118654752440))
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_gelu[(Mrows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
