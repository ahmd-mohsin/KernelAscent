import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 446
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_rms_relu_kernel(
    X_ptr, W_ptr, Y_ptr,
    D: tl.constexpr,
    stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), matching F.gelu default
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # RMS norm over the row (float32 accumulation)
    ss = tl.sum(tl.where(mask, g * g, 0.0), axis=0)
    rms = tl.math.rsqrt(ss / D + EPS)

    # normalize, cast to bf16 (matches .to(x.dtype)), then multiply by bf16 weight
    n_bf16 = (g * rms).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)

    y = (n_bf16.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # ReLU
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_rms_relu_kernel[(m,)](
            x, self.rms1_w, y,
            d,
            x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
