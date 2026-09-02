import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 642
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _relu_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    # relu in bf16 (equivalent to relu in fp32 after cast)
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    normed_bf16 = (xf * inv).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = (normed_bf16.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _relu_rmsnorm_kernel[(Mrows,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
