import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 383
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _relu_rmsnorm_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y

    # Pass 1: sum of squares of relu(x) in fp32
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        acc += x * x
    ss = tl.sum(acc, axis=0)
    rstd = tl.math.rsqrt(ss / N + eps)

    # Pass 2: normalize, cast to bf16, multiply by weight (fp32 math), store bf16
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        xn = (x * rstd).to(tl.bfloat16).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = (xn * w).to(tl.bfloat16)
        tl.store(y_ptr + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        w = self.rms1_w
        BLOCK = 1024
        grid = (Mrows,)
        _relu_rmsnorm_kernel[grid](
            x, w, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
