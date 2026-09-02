import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 463
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _rmsnorm_bias_kernel(
    X, W, B, Y,
    N,
    stride_xm, stride_ym,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (x * rstd).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = xn * w + b
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core matmul
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_bias_kernel[(Mrows,)](
            h, self.rms1_w, self.b2, y,
            N,
            h.stride(0), y.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
