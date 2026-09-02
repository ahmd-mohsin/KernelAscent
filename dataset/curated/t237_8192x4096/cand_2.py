import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 237
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_ln_gelu_ln_relu(
    X, Y, G1, B1, G3, B3,
    stride_x, stride_y,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + cols).to(tl.float32)

    # LayerNorm 1
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(G1 + cols).to(tl.float32)
    b1 = tl.load(B1 + cols).to(tl.float32)
    h = xc * rstd * g1 + b1
    # round to fp16 to match reference intermediate precision
    h = h.to(tl.float16).to(tl.float32)

    # GELU (exact, erf)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    h = h * 0.5 * (1.0 + tl.math.erf(h * INV_SQRT2))
    h = h.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(h, axis=0) / N
    hc = h - mean2
    var2 = tl.sum(hc * hc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g3 = tl.load(G3 + cols).to(tl.float32)
    b3 = tl.load(B3 + cols).to(tl.float32)
    o = hc * rstd2 * g3 + b3
    o = o.to(tl.float16).to(tl.float32)

    # scale + relu
    o = o * 1.4996
    o = o.to(tl.float16).to(tl.float32)
    o = tl.maximum(o, 0.0)

    tl.store(Y + row * stride_y + cols, o.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _fused_ln_gelu_ln_relu[(Mrows,)](
            x, y, self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            x.stride(0), y.stride(0),
            N=N, BLOCK=N,
            num_warps=8,
        )
        return y
