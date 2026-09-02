import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 965
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_ln_gelu_kernel(
    X, W, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # x * 1.311 (computed in fp32, rounded back to bf16 like the eager op)
    xf = x.to(tl.float32) * 1.311
    xf = xf.to(tl.bfloat16).to(tl.float32)

    # layer norm statistics in fp32
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * w + b
    # layer_norm output is cast to bf16
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU in fp32
    g = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    out = (g * 1.4814).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_gelu_kernel[(Mrows,)](
            x2d, self.ln1_g, self.ln1_b, y,
            x2d.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
