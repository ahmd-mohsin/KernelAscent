import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 551
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, stride,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, matching x.float() path)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # fp16 arithmetic to match: (..).to(dtype) * w + b
    y = xn * w + b

    # softmax #1: compute in fp32 (as PyTorch does for half), output fp16
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m1 = tl.max(yf, axis=0)
    e1 = tl.exp(yf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16)

    # softmax #2
    pf = p1.to(tl.float32)
    pf = tl.where(mask, pf, float('-inf'))
    m2 = tl.max(pf, axis=0)
    e2 = tl.exp(pf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.float16)

    tl.store(out_ptr + row * stride + cols, p2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(rows,)](
            x, self.rms0_w, self.b1, out, n, x.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
