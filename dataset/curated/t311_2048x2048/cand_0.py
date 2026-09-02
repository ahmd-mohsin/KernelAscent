import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 311
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B0, B2, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    # x + b0  (round to bf16 like reference elementwise op)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)
    # x * 1.357
    x = (x * 1.357).to(tl.bfloat16).to(tl.float32)
    # x + b2
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, output rounded to bf16
    x_max = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    ex = tl.exp(x - x_max)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    sm = (ex / denom).to(tl.bfloat16).to(tl.float32)

    # layer norm in fp32
    mean = tl.sum(tl.where(mask, sm, 0.0), axis=0) / N
    diff = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (sm - mean) * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.b0, self.b2, self.ln4_g, self.ln4_b, y,
            x2.stride(0), y.stride(0),
            N, BLOCK=BLOCK, EPS=1e-5,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
