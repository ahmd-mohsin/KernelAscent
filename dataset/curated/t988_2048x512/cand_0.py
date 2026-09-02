import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 988
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _rms_softmax_kernel(
    X, W, Y,
    N,
    stride_xm,
    stride_ym,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    # cast back to fp16, multiply by weight in fp16 (matches reference)
    xn = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    y16 = xn * w  # fp16 multiply

    # softmax in fp32 (matches torch half softmax which accumulates in float)
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_softmax_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N,
            x.stride(0),
            y.stride(0),
            EPS=1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
