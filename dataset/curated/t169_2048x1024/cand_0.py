import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 169
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_gelu_rms_bias_kernel(
    X, W, B2, B3, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then cast to fp16 (matches F.gelu on half)
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    g16 = g.to(tl.float16)

    # RMSNorm in fp32 on the fp16 gelu output
    gf = g16.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    n16 = (gf * inv).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0)

    # fp16 arithmetic order: (n*w) + b2, then + b3
    out = (n16 * w + b2) + b3
    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xg = F.gelu(x)
            _xf = xg.float()
            xn = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xg.dtype) * self.rms1_w
            return xn + self.b2 + self.b3

        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_gelu_rms_bias_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, self.b3, y,
            D=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
