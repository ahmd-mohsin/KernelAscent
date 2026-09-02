import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 883
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _fused_kernel(X, Y, G, B,
                  stride_xm, stride_ym,
                  N, eps, scale,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 math, round to fp16 like PyTorch output)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    x = (e1 / s1).to(tl.float16).to(tl.float32)

    # softmax 2
    x_for_max = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x_for_max, axis=0)
    e2 = tl.exp(x - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    x = (e2 / s2).to(tl.float16).to(tl.float32)

    # layer norm (fp32 math)
    x_m = tl.where(mask, x, 0.0)
    mean = tl.sum(x_m, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # scale (fp32 opmath, round to fp16)
    y = (y * scale).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, y, self.ln2_g, self.ln2_b,
            x.stride(0), y.stride(0),
            N, 1e-5, 1.0249,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
