import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 163
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_ln_kernel(
    X, W, B, Y,
    stride_x, stride_y,
    N, eps,
    S1, S2, S3,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # replicate the two fp16 scalings with intermediate rounding
    xf = x.to(tl.float32) * S1
    xh = xf.to(tl.float16)
    xf = xh.to(tl.float32) * S2
    xh = xf.to(tl.float16)
    xf = xh.to(tl.float32)

    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xf - mean) * rstd * w + b
    yh = y.to(tl.float16)
    # relu in fp16
    yh = tl.maximum(yh, 0.0)
    # final scale (fp16 mul semantics via fp32 with rounding)
    out = (yh.to(tl.float32) * S3).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_kernel[(Mrows,)](
            x, self.ln2_g, self.ln2_b, y,
            x.stride(0), y.stride(0),
            N, 1e-5,
            1.1762, 1.0706, 1.2325,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
