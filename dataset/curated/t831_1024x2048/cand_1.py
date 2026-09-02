import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 831
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _relu_rmsnorm_kernel(
    X, W, Y,
    stride_x, stride_y,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # ReLU
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)
    # RMS
    mean_sq = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    xn = (xf * rstd).to(Y.dtype.element_ty)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _relu_rmsnorm_kernel[(Mrows,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
