import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 461
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_scale_rmsnorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    # match reference: bf16 multiply (round to bf16), then upcast to fp32
    x_scaled = (x.to(tl.float32) * scale).to(tl.bfloat16)
    xf = x_scaled.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    normed = (xf * inv).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    y = normed * w  # bf16 mul
    y = y + b       # bf16 add

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS tensor-core matmul
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_scale_rmsnorm_kernel[(Mrows,)](
            x, self.rms2_w, self.b3, y,
            N, x.stride(0), y.stride(0),
            1e-6, 1.0961,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
