import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 948
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _rmsnorm_relu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, eps,
    stride_x, stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptrs = X_ptr + row * stride_x + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    xn = (x * rstd).to(tl.float16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = xn * w
    y = tl.maximum(y, 0.0)

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 4096 else 4
        _rmsnorm_relu_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N, 1e-6,
            x.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
