import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 521
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, W, Y, stride_x, stride_y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)

    # x * 1.405 (bf16 elementwise, computed in fp32 then rounded to bf16)
    t = (x.to(tl.float32) * 1.405).to(tl.bfloat16)
    # + b1 (bf16)
    t = (t.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    # relu
    tf = t.to(tl.float32)
    tf = tl.where(tf > 0.0, tf, 0.0)

    # softmax in fp32 (matching PyTorch's fp32 accumulation for bf16 input)
    tf = tl.where(mask, tf, float('-inf'))
    m = tl.max(tf, axis=0)
    e = tl.exp(tf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)  # softmax output rounded to bf16

    # RMSNorm: cast to fp32, mean of squares, rsqrt, scale, cast to bf16
    xf = sm.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    normed = (xf * inv).to(tl.bfloat16)

    # multiply by weight (bf16 elementwise via fp32)
    w = tl.load(W + offs, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.b1, self.rms4_w, y,
            x.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
