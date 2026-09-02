import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 422
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _relu_scale_rmsnorm_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    # relu in fp16
    x = tl.maximum(x, 0.0)
    # multiply by 1.0221 with fp16 rounding (matches x * 1.0221 on fp16 tensor)
    x = (x.to(tl.float32) * 1.0221).to(tl.float16)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    y = (xf * inv).to(tl.float16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # fp16
    out = y * w  # fp16 multiply
    tl.store(Out_ptr + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _relu_scale_rmsnorm_kernel[(m,)](
            y, self.rms3_w, out,
            y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
