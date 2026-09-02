import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 370
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _rmsnorm_bias_kernel(
    X, W, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    # match reference: (xf * rsqrt).to(bf16) * w  + b
    xn = (xf * inv).to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    y = xn * w + b
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 tensor-core GEMM
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_bias_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, y,
            x.stride(0), y.stride(0),
            N=N, EPS=1e-6, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
