import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 425
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_scale_rms_bias(
    X, W, B, Y,
    D: tl.constexpr,
    stride_x, stride_y,
    eps,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    # x = x * 1.3008 performed in fp16 (match torch fp16 elementwise mul)
    xs = (x.to(tl.float32) * scale).to(tl.float16)

    xf = xs.to(tl.float32)
    ssq = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0)
    rstd = 1.0 / tl.sqrt(ssq / D + eps)

    normed = (xf * rstd).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)  # fp16

    out = normed * w + b  # fp16 arithmetic, matches torch
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_scale_rms_bias[(m,)](
            x, self.rms1_w, self.b2, y,
            d,
            x.stride(0), y.stride(0),
            1e-6,
            1.3008,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
