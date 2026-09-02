import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 907
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_relu_scale_rmsnorm(
    X, W, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # relu in bf16
    x = tl.maximum(x, 0.0)
    # x * 1.0822 with bf16 rounding
    x = (x.to(tl.float32) * 1.0822).to(tl.bfloat16)
    # x * 1.1211 with bf16 rounding
    x = (x.to(tl.float32) * 1.1211).to(tl.bfloat16)

    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)

    normed = (xf * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = normed * w

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_relu_scale_rmsnorm[(m,)](
            x2, self.rms3_w, y,
            x2.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
