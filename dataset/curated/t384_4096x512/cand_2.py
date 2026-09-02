import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 384
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_gelu_relu_rms2_kernel(
    X, W3, W4, Out,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf) computed in fp32, cast to fp16 (matches PyTorch opmath)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16)
    # ReLU
    g = tl.maximum(g, 0.0)

    # first RMSNorm
    xf = g.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    y = (xf * tl.math.rsqrt(ms + eps)).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w3.to(tl.float32)).to(tl.float16)

    # second RMSNorm
    yf = y.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    z = (yf * tl.math.rsqrt(ms2 + eps)).to(tl.float16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w4.to(tl.float32)).to(tl.float16)

    tl.store(Out + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_relu_rms2_kernel[(Mrows,)](
            h, self.rms3_w, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
