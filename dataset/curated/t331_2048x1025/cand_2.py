import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 331
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_rms_gelu_softmax(
    X, W1, W3, W4, OUT,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = X + row * stride

    x = tl.load(base + offs, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    y = (xf * r).to(tl.float16) * w1  # fp16 mul like PyTorch

    # GELU (exact, computed in fp32 like PyTorch opmath, cast back to fp16)
    yf = y.to(tl.float32)
    g = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    y = g.to(tl.float16)

    # RMSNorm 3
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    y = (yf * r).to(tl.float16) * w3

    # RMSNorm 4
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0)
    y = (yf * r).to(tl.float16) * w4

    # Softmax (fp32 accumulation, fp16 output)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    mmax = tl.max(yf, axis=0)
    e = tl.exp(yf - mmax)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores on A100)
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_softmax[(Mrows,)](
            x, self.rms1_w, self.rms3_w, self.rms4_w, out,
            N, x.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
