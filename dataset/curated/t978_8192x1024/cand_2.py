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
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(ms + 1e-6)
    x_rms = xf * inv_rms
    # round to bf16 (matches .to(x.dtype))
    x_rms = x_rms.to(tl.bfloat16)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> bf16 (exact fp32 product rounded to bf16)
    t = (x_rms.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm in fp32
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, 0.0)
    mean = tl.sum(tf, axis=0) / N
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = ((tf - mean) * inv_std * g + b).to(tl.bfloat16)

    # GELU (exact, erf) in fp32
    lf = ln.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * lf * (1.0 + tl.math.erf(lf * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_gelu[(M_,)](
            x, self.rms0_w, self.ln1_g, self.ln1_b, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
