import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 269
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _rms_relu_bias_kernel(
    X, W, B, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    xn = (xf * rstd).to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    y = xn * w
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)
    y = y + b

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _rms_relu_bias_kernel[(m,)](
            x, self.rms1_w, self.b3, y,
            n, x.stride(0), y.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y
