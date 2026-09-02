import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 844
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _ln_bias_relu_kernel(
    X, OUT, G, B, B2, B3,
    N, stride_x, stride_o,
    EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    y = y.to(tl.bfloat16)

    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)

    y = (y.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
    y = tl.maximum(y, 0.0)
    y = (y.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        Mr, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _ln_bias_relu_kernel[(Mr,)](
            h, out, self.ln1_g, self.ln1_b, self.b2, self.b3,
            N, h.stride(0), out.stride(0),
            EPS=1e-5, SCALE=1.4218,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
