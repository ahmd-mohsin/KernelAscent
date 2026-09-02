import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 271
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B1, W, Y, D_dim, stride_x, stride_y, EPS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_dim

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # gelu 1 (compute in fp32, round to fp16 like PyTorch elementwise)
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.float16)
    # add bias (fp16)
    xb = g1 + b1
    # gelu 2
    xf = xb.to(tl.float32)
    g2 = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g2 = g2.to(tl.float16)
    # rmsnorm in fp32
    gf = g2.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / D_dim
    inv = 1.0 / tl.sqrt(ms + EPS)
    normed = (gf * inv).to(tl.float16)
    out = normed * w
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, Dcols)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(x2.shape[0],)](
            x2, self.b1, self.rms3_w, y,
            Dcols, x2.stride(0), y.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return y.view_as(x)
