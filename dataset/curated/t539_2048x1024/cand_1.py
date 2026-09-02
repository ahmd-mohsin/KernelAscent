import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 539
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def fused_bias_scale_gelu_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_xm, stride_ym,
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # bias + scale
    x = (x + b) * scale

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # softmax (numerically stable, fp32 accumulation)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    x = x - row_max
    ex = tl.exp(x)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    y = ex / denom

    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        fused_bias_scale_gelu_softmax_kernel[(Mrows,)](
            x, self.b0, y,
            x.stride(0), y.stride(0),
            N, 1.1541,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
