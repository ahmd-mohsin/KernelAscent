import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 45
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_bias_rms_kernel(
    X, B1, B2, W, B4, Out,
    N, stride_x, stride_o,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # emulate fp16 sequential adds
    x = (x + b1).to(tl.float16)
    x = (x + b2).to(tl.float16)

    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    y = (xf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)

    y = (y * w).to(tl.float16)
    y = (y + b4).to(tl.float16)

    tl.store(Out + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_bias_rms_kernel[(Mrows,)](
            x, self.b1, self.b2, self.rms3_w, self.b4, out,
            N, x.stride(0), out.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
