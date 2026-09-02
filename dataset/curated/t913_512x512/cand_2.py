import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 913
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0)  # bf16

    # scale 1: bf16 rounding as in reference
    x = (x.to(tl.float32) * 1.3419).to(tl.bfloat16)
    # scale 2
    x = (x.to(tl.float32) * 1.3693).to(tl.bfloat16)

    # exact GELU (computed in fp32, rounded back to bf16 like PyTorch opmath)
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = g.to(tl.bfloat16)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)  # bf16
    y = xn * w  # bf16 multiply

    tl.store(Y + row * D + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, D_)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(D_)
        _fused_kernel[(rows,)](x2, self.rms3_w, y, D_, BLOCK=BLOCK, num_warps=4)
        return y.view_as(x)
