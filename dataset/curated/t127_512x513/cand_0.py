import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 127
M, D, DT = 512, 513, torch.bfloat16


@triton.jit
def _rms_relu_bias_kernel(
    X, W, B, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # match: (xf * rsqrt).to(bf16) * w  -> bf16 mul with fp32 opmath, round to bf16
    t = (xf * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (t * w).to(tl.bfloat16)

    # relu in bf16
    y = tl.maximum(y, 0.0)

    # bf16 add with fp32 opmath
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        M_, N = x.shape
        x = x.contiguous()
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_relu_bias_kernel[(M_,)](
            x, self.rms1_w, self.b3, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
