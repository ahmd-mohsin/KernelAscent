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
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y

    # Pass 1: sum of squares of relu(x) in fp32
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        acc += x * x
    ssum = tl.sum(acc, axis=0)
    inv = tl.math.rsqrt(ssum / N + eps)

    # Pass 2: normalize, cast to bf16, multiply weight in bf16
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        y = (x * inv).to(tl.bfloat16)
        w = tl.load(W + cols, mask=mask, other=0.0)
        tl.store(y_ptr + cols, y * w, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        w = self.rms1_w
        BLOCK_N = 2048
        _relu_rmsnorm_kernel[(m,)](
            x, w, y,
            n, x.stride(0), y.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
