import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 659
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _relu_rmsnorm_kernel(
    X, W, Y,
    D_: tl.constexpr,
    stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)  # relu (idempotent, so applying once == twice)

    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + eps)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xf * inv).to(tl.float16) * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D_)
        _relu_rmsnorm_kernel[(M_,)](
            x, self.rms2_w, y,
            D_,
            x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
