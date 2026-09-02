import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 551
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_kernel(
    X, W, B, OUT,
    n_cols,
    stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (float32 math, matching reference)
    ms = tl.sum(x * x, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + eps)
    xn = (x * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    y = xn * w        # fp16 multiply, as in reference
    y = y + b         # fp16 add

    # softmax #1 (fp32 accumulation, fp16 output)
    yf = tl.where(mask, y.to(tl.float32), float('-inf'))
    m1 = tl.max(yf, axis=0)
    e1 = tl.exp(yf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    o1 = (e1 / s1).to(tl.float16)

    # softmax #2 (fp32 accumulation, fp16 output)
    zf = tl.where(mask, o1.to(tl.float32), float('-inf'))
    m2 = tl.max(zf, axis=0)
    e2 = tl.exp(zf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    o2 = (e2 / s2).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, o2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(n_rows,)](
            x, self.rms0_w, self.b1, out,
            n_cols,
            x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
