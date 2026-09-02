import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 139
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, stride_x, stride_y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, scaled by fp16 weight)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    h = (xn * w).to(tl.float16)

    # softmax 1 (fp32 math, fp16 output)
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, float('-inf'))
    m1 = tl.max(hf, axis=0)
    e1 = tl.exp(hf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16)

    # softmax 2 (fp32 math, fp16 output)
    pf = p1.to(tl.float32)
    pf = tl.where(mask, pf, float('-inf'))
    m2 = tl.max(pf, axis=0)
    e2 = tl.exp(pf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.float16)

    # exact GELU (erf), fp32 math, fp16 output
    g = p2.to(tl.float32)
    out = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    tl.store(Y + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return F.gelu(x)

        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms0_w, y,
            x.stride(0), y.stride(0),
            D=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
