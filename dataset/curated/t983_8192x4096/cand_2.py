import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 983
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _rmsnorm_relu_kernel(
    X, W, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / D
    inv = tl.math.rsqrt(ms + EPS)

    # normalize in fp32, round to bf16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.bfloat16)

    # bf16 * bf16 (single rounding, same as PyTorch bf16 mul)
    y = xn * w
    y = tl.maximum(y, y - y)  # relu via max(y, 0) in bf16
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _rmsnorm_relu_kernel[(m,)](
            x, self.rms0_w, y,
            x.stride(0), y.stride(0),
            D=d, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
