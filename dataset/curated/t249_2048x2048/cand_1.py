import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 249
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMS norm (mean of squares over the row, in fp32)
    ms = tl.sum(xf * xf, axis=0) / N
    rs = tl.math.rsqrt(ms + EPS)

    # (xf * rsqrt) rounded to bf16 (matches .to(x.dtype))
    y = (xf * rs).to(tl.bfloat16)

    # multiply by weight in bf16 semantics (compute fp32, round to bf16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w).to(tl.bfloat16)

    # scalar scale, bf16 semantics
    y = (y.to(tl.float32) * SCALE).to(tl.bfloat16)

    # exact GELU (erf), computed in fp32 then rounded to bf16
    yf = y.to(tl.float32)
    g = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N, x.stride(0), y.stride(0),
            EPS=1e-6, SCALE=1.2502,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
