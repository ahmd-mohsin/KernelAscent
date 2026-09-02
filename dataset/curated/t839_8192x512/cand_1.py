import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 839
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, stride_x, stride_y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, cast result to fp16 to match PyTorch)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    x1 = (e1 / s1).to(tl.float16).to(tl.float32)
    x1 = tl.where(mask, x1, float('-inf'))

    # softmax 2
    m2 = tl.max(x1, axis=0)
    e2 = tl.exp(x1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    x2 = (e2 / s2).to(tl.float16)

    # RMSNorm in fp32
    xf = x2.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * rrms).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w  # fp16 multiply, matching reference

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms2_w, y,
            x.stride(0), y.stride(0),
            D=Dcols, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
