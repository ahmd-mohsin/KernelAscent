import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 463
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _rms_scale_bias_kernel(
    X, W, B, Out,
    N,
    stride_xm,
    stride_om,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    mean_sq = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # normalize in fp32, cast to fp16, then multiply by weight (match reference order)
    xn = (xf * rstd).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    y = xn * w + b
    tl.store(Out + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_scale_bias_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, out,
            N,
            x.stride(0),
            out.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
