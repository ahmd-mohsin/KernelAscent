import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 497
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_scale_relu_ln_kernel(
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
    # scale by 1.15 in fp32 then round to bf16 (matches eager x * 1.15 on bf16)
    x = (x.to(tl.float32) * 1.15).to(tl.bfloat16)
    # relu
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # layernorm statistics in fp32
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xf - mean) * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_relu_ln_kernel[(Mrows,)](
            x, self.ln2_g, self.ln2_b, y,
            x.stride(0), y.stride(0),
            N, 1e-5, BLOCK,
            num_warps=8,
        )
        return y
