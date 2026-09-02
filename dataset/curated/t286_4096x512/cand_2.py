import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 286
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _scale_ln_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # emulate fp16 multiply (rounded to fp16), then upcast to fp32 like layer_norm does
    xs = (x.to(tl.float32) * SCALE).to(tl.float16).to(tl.float32)

    mean = tl.sum(xs, axis=0) / N
    diff = tl.where(mask, xs - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xs - mean) * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _scale_ln_kernel[(M_,)](
            x, self.ln1_g, self.ln1_b, y,
            x.stride(0), y.stride(0),
            N=N, SCALE=1.2664, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
