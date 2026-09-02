import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 794
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_gelu_softmax(
    X, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)

    # x = (x + b0) * 1.4147   (match bf16 intermediate rounding)
    x = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) * 1.4147).to(tl.bfloat16)

    # softmax (fp32 accumulation, round to bf16 like torch)
    xf = tl.where(mask, x.to(tl.float32), float('-inf'))
    m1 = tl.max(xf, axis=0)
    e1 = tl.exp(xf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = (e1 / s1).to(tl.bfloat16)

    # exact GELU: 0.5*x*(1+erf(x/sqrt(2))), fp32 compute, bf16 rounding
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # second softmax
    gf = tl.where(mask, g.to(tl.float32), float('-inf'))
    m2 = tl.max(gf, axis=0)
    e2 = tl.exp(gf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = x + self.b0
            x = x * 1.4147
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, d = x.shape[0] * (x.numel() // (x.shape[-1] * x.shape[0])) if x.dim() > 2 else x.shape[0], x.shape[-1]
        rows = x.numel() // d
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_softmax_gelu_softmax[(rows,)](
            x, self.b0, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
