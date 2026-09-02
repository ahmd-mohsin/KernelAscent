import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 979
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_post_kernel(
    X, B1, W2, W3, G4, Be4, Out,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)                 # fp16

    # bias add in fp16 (matches x + b1 half arithmetic)
    x = x + b1

    # RMSNorm 1: compute in fp32, cast to fp16, multiply by fp16 weight
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)  # fp16
    x = (xf * inv).to(tl.float16) * w2

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)  # fp16
    x = (xf * inv).to(tl.float16) * w3

    # LayerNorm: fp32 compute, fp32 affine, cast to fp16
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    be = tl.load(Be4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd) * g + be

    tl.store(Out + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            h, self.b1, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
