import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 343
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_relu_rms_relu_scale(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    # relu (applied twice == once)
    xf = tl.maximum(xf, 0.0)

    # rmsnorm over row
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + EPS)

    # match reference: (xf * rsqrt).to(fp16) * w  -> cast before weight mul
    xn = (xf * rstd).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    # relu then scale (in fp16, matching reference dtype semantics)
    y = tl.maximum(y, 0.0)
    y = y * SCALE

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4
        _fused_relu_rms_relu_scale[(Mrows,)](
            x, self.rms2_w, y,
            x.stride(0), y.stride(0),
            N=N, EPS=1e-6, SCALE=1.1395,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
