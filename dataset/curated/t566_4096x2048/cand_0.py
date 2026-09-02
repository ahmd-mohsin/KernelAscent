import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 566
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_rms_gelu_kernel(
    X, W, B2, B4, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS norm (mean of squares in fp32)
    ms = tl.sum(x * x, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + 1e-6)

    # x = (xf * rsqrt).to(fp16)
    y = (x * rs).to(tl.float16)

    # * rms1_w  (half*half computed in fp32, rounded to fp16 like PyTorch opmath)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # + b2
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)

    # gelu (exact, computed in fp32, cast back to fp16)
    yf = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    yf = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    y = yf.to(tl.float16)

    # + b4
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b4.to(tl.float32)).to(tl.float16)

    # gelu
    yf = y.to(tl.float32)
    yf = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    y = yf.to(tl.float16)

    tl.store(Out + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        grid = (m,)
        _fused_rms_gelu_kernel[grid](
            x, self.rms1_w, self.b2, self.b4, out,
            x.stride(0), out.stride(0),
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
