import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 354
M, D, DT = 1024, 1025, torch.float16


@triton.jit
def _ln_gelu_relu_kernel(
    X, G, B, Y,
    stride_xm,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # mean / variance in fp32 (matches PyTorch fp16 layer_norm internals)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b

    # cast to fp16 (layer_norm output dtype), then gelu in fp32 math
    y = y.to(tl.float16).to(tl.float32)

    # exact GELU (erf)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    gel = gel.to(tl.float16).to(tl.float32)

    # relu
    out = tl.maximum(gel, 0.0)

    tl.store(Y + row * stride_xm + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _ln_gelu_relu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, y,
            h.stride(0),
            N, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
