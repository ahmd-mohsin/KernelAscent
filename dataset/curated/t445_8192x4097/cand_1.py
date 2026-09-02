import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 445
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _rmsnorm_relu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    # match reference: (xf * rsqrt).to(bf16) * w  -> then relu
    xn = (x * rstd).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = xn * w
    zero = tl.zeros(y.shape, dtype=y.dtype)
    y = tl.maximum(y, zero)

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x @ self.W1
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_relu_kernel[(Mrows,)](
            x, self.rms2_w, y,
            N,
            x.stride(0), y.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
