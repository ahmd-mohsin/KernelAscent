import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 302
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X_ptr, W1_ptr, W2_ptr, Out_ptr,
    N, stride_row,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0)  # fp16
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)  # fp16
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0)  # fp16

    # First RMSNorm
    xf = x.to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + eps)
    y = (xf * r1).to(tl.float16) * w1  # fp16 multiply, matches PyTorch

    # Second RMSNorm
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + eps)
    z = (yf * r2).to(tl.float16) * w2

    tl.store(Out_ptr + row * stride_row + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _double_rmsnorm_kernel[(Mrows,)](
            x, self.rms1_w, self.rms2_w, out,
            N, x.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
