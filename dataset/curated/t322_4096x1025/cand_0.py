import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 322
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_relu_ln_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Load row, ReLU (compute in fp32, matching PyTorch opmath for bf16)
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # LayerNorm statistics in fp32
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + LN_EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # layer_norm output rounds to bf16
    y = y.to(tl.bfloat16).to(tl.float32)

    # scalar multiply (fp32 opmath, then round to bf16)
    y = y * SCALE
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    rrms = tl.math.rsqrt(ms + RMS_EPS)
    z = (y * rrms).to(tl.bfloat16).to(tl.float32)  # cast to bf16 as in reference

    # multiply by rms weight (fp32 opmath, round to bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        M_rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_ln_rms_kernel[(M_rows,)](
            x2d, self.ln1_g, self.ln1_b, self.rms3_w, y,
            N, x2d.stride(0), y.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6, SCALE=1.4215,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
