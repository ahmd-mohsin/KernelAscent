import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 384
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_gelu_relu_double_rmsnorm(
    X, W3, W4, OUT,
    N, stride_row,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_row + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # GELU (erf-based, computed in fp32, stored back to fp16 like PyTorch)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # ReLU on fp16
    zero16 = tl.zeros_like(g16)
    r16 = tl.maximum(g16, zero16)

    # First RMSNorm
    rf = r16.to(tl.float32)
    ms1 = tl.sum(tl.where(mask, rf * rf, 0.0), axis=0) / N
    inv1 = tl.math.rsqrt(ms1 + EPS)
    y1_16 = (rf * inv1).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    y1 = y1_16 * w3  # fp16 multiply, matches reference

    # Second RMSNorm
    yf = y1.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    inv2 = tl.math.rsqrt(ms2 + EPS)
    y2_16 = (yf * inv2).to(tl.float16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    out = y2_16 * w4

    tl.store(OUT + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core GEMM
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_relu_double_rmsnorm[(m,)](
            h, self.rms3_w, self.rms4_w, out,
            n, h.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
