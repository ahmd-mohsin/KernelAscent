import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 579
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_rms_softmax_gelu(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm (fp32 math, then cast to fp16 and multiply by fp16 weight)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * r).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)  # fp16
    y16 = xn * w  # fp16 multiply, matches reference

    # Softmax in fp32, output fp16
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # relu (identity on softmax output, kept for equivalence) + exact GELU (erf)
    smr = tl.maximum(sm, 0.0)
    t = smr.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * t * (1.0 + tl.math.erf(t * INV_SQRT2))
    out = g.to(tl.float16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_gelu[(Mrows,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
