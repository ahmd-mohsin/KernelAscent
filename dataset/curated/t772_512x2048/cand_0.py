import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 772
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _ln_relu_softmax_kernel(
    X, G, B, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    # match fp16 layer_norm output precision: cast to fp16 then back
    y = ((xc * rstd) * g + b).to(tl.float16).to(tl.float32)

    # ReLU
    y = tl.maximum(y, 0.0)

    # Softmax
    y = tl.where(mask, y, float('-inf'))
    ymax = tl.max(y, axis=0)
    e = tl.exp(y - ymax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _ln_relu_softmax_kernel[(m,)](
            h, self.ln1_g, self.ln1_b, y,
            h.stride(0), y.stride(0),
            n, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
