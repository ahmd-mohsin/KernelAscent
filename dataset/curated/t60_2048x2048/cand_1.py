import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 60
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_relu_bias_rmsnorm(
    X, B1, W, Y,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B1 + cols, mask=mask, other=0.0)

    # relu then add bias in bf16 (matches reference dtype behavior)
    x = tl.maximum(x, 0.0)
    x = (x + b).to(tl.bfloat16)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    xn = (xf * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = xn * w

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_bias_rmsnorm[(Mrows,)](
            x, self.b1, self.rms2_w, y,
            x.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
