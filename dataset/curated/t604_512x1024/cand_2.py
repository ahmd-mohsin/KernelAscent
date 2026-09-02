import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 604
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu (in input dtype, exact)
    x = tl.maximum(x, 0.0)

    # rmsnorm in float32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = tl.math.rsqrt(ms + 1e-6)
    normed = (xf * inv).to(tl.bfloat16)

    # * weight (bf16*bf16 -> computed in fp32 opmath, rounded to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # * scalar (computed in fp32 opmath, rounded to bf16)
    y = (y.to(tl.float32) * 1.2572).to(tl.bfloat16)

    # final relu
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
