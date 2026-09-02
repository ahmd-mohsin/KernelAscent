import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 718
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_ln_gelu_ln_kernel(
    X, Y, G0, B0, G2, B2,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 0 (fp32 math like PyTorch)
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + EPS)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    h = d0 * rstd0 * g0 + b0
    # cast to bf16 to match PyTorch intermediate storage
    h = h.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    h = 0.5 * h * (1.0 + tl.math.erf(h * INV_SQRT2))
    h = h.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    d2 = tl.where(mask, h - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g2 + b2

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_ln_gelu_ln_kernel[(Mrows,)](
            x2d, y,
            self.ln0_g, self.ln0_b, self.ln2_g, self.ln2_b,
            x2d.stride(0), y.stride(0),
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
