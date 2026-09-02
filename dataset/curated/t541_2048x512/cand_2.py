import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 541
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_rmsnorm_kernel(
    X, W, B, Y,
    stride_x, stride_y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    xn = (xf * rstd).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float16)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float16)

    y = xn * w
    y = y + b
    y = y * scale.to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rmsnorm_kernel[(m,)](
            x, self.rms0_w, self.b1, y,
            x.stride(0), y.stride(0),
            n, 1e-6, 1.4119,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
