import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 406
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X, W, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    # round to bf16 (as torch.softmax output is bf16), then re-read as fp32
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(yf * yf, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)
    a = (yf * r).to(tl.bfloat16).to(tl.float32)

    # multiply by weight (bf16 op with fp32 opmath, rounded to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (a * w).to(tl.bfloat16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_kernel[(Mrows,)](
            x, self.rms2_w, out,
            x.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK, EPS=1e-6,
            num_warps=4,
        )
        return out
