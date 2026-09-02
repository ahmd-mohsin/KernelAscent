import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 756
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _rms_scale_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    mean_sq = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(mean_sq + EPS)

    xn = (xf * rrms).to(x.dtype)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = xn * w * SCALE

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _rms_scale_kernel[(m,)](
            x, self.rms1_w, y,
            n, x.stride(0), y.stride(0),
            EPS=1e-6, SCALE=1.1814,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
