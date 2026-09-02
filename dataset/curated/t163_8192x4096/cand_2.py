import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 163
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_scale_ln_relu_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # replicate the two scalar multiplies (opmath fp32, stored back to fp16 each step)
    x = (x.to(tl.float32) * 1.1762).to(tl.float16)
    x = (x.to(tl.float32) * 1.0706).to(tl.float16)

    xf = x.to(tl.float32)

    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xf - mean) * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)  # layer_norm output rounded to fp16
    y = tl.maximum(y, 0.0)               # relu
    y = y * 1.2325                       # final scale (opmath fp32)

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_ln_relu_kernel[(M_,)](
            x, self.ln2_g, self.ln2_b, y,
            x.stride(0), y.stride(0),
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
