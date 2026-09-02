import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 978
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_rms_ln_gelu(
    X, W_RMS, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ----- RMSNorm (computed in fp32, output rounded to bf16, then * weight in fp32, rounded to bf16) -----
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + EPS_RMS)
    xr = (x * rstd).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0).to(tl.float32)
    xr = (xr * w).to(tl.bfloat16).to(tl.float32)

    # ----- LayerNorm (fp32 internal math on bf16 input, bf16 output) -----
    mean = tl.sum(tl.where(mask, xr, 0.0), axis=0) / N
    diff = tl.where(mask, xr - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xr - mean) * inv * g + b).to(tl.bfloat16).to(tl.float32)

    # ----- GELU (erf form, fp32 internal, bf16 output) -----
    out = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_rms_ln_gelu[(Mrows,)](
            x2, self.rms0_w, self.ln1_g, self.ln1_b, y,
            x2.stride(0), y.stride(0),
            N=N,
            EPS_RMS=1e-6,
            EPS_LN=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
