import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 478
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_relu_rms_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    # relu (applying twice == once)
    zero16 = tl.zeros_like(x)
    x = tl.maximum(x, zero16)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    y = (xf * r).to(tl.float16)  # cast back to fp16 first (matches reference)
    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    y = y * w  # fp16 multiply
    y = tl.maximum(y, tl.zeros_like(y))  # final relu

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_relu_rms_kernel[(Mrows,)](
            x, self.rms3_w, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y
