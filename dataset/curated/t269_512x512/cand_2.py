import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 269
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _rms_relu_bias_kernel(
    X, W, B, Y,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # normalized, rounded to bf16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 elementwise mul: compute in fp32, round once to bf16
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # relu (no rounding change)
    y = tl.maximum(y, tl.zeros_like(y))

    b = tl.load(B + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_xm + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _rms_relu_bias_kernel[(m,)](
            x, self.rms1_w, self.b3, y,
            x.stride(0),
            N=n,
            BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return y
