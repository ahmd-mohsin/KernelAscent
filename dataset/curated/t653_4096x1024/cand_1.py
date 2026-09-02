import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 653
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W0, W1, B, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm 0
    ms = tl.sum(xf * xf, axis=0) / D_
    r = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * r).to(tl.bfloat16).to(tl.float32)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = (xn * w0).to(tl.bfloat16)

    # RMSNorm 1
    xf = x1.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    r = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * r).to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = (xn * w1).to(tl.bfloat16)

    # GELU (exact, erf) x2 - compute in fp32, round to bf16 between ops
    xf = x2.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    xf = g.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # bias add in bf16 semantics (fp32 add of bf16 operands is exact, then round)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (g.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.gelu(x)
            x = F.gelu(x)
            return x + self.b4

        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms0_w, self.rms1_w, self.b4, y,
            x.stride(0), y.stride(0),
            D_=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
