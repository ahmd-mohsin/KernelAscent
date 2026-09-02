import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 834
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _relu_rmsnorm_kernel(
    X, W, Y,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)  # relu

    mean_sq = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xf * rstd).to(Y.dtype.element_ty) * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_rmsnorm_kernel[(M_,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
