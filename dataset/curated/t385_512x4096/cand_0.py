import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 385
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _rmsnorm_relu_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    xn = (x * rstd).to(Y.dtype.element_ty)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    zero = tl.zeros_like(y)
    y = tl.where(y > zero, y, zero)
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        M_, N = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 4096 else 4
        _rmsnorm_relu_kernel[(M_,)](
            x2d, self.rms0_w, y,
            x2d.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
