import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 808
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X, W, OUT,
    stride_xm, stride_om,
    N, EPS, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # rmsnorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    y = (x * inv).to(tl.float16)

    # * weight (fp16 storage, fp32 opmath, round to fp16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # relu
    y = tl.maximum(y, tl.zeros_like(y))

    # * scalar (fp32 opmath, round to fp16)
    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    # softmax in fp32
    yf = tl.where(mask, y.to(tl.float32), float('-inf'))
    mval = tl.max(yf, axis=0)
    e = tl.exp(yf - mval)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x, self.rms1_w, out,
            x.stride(0), out.stride(0),
            n, 1e-6, 1.3505,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
