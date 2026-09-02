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
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    # relu in input dtype (bf16), matching reference: relu applied before float cast
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    mean_sq = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(mean_sq + eps)

    # (xf * inv) cast back to bf16, then multiplied by weight (bf16 * bf16)
    normed = (xf * inv).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = normed * w

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_rmsnorm_kernel[(Mrows,)](
            h, self.rms2_w, y,
            N, h.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
