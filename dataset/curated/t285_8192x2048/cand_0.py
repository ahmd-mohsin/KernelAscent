import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 285
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _bias_rmsnorm_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # x + b in bf16 (match reference: bias add in bf16, then cast to float)
    xb = (x + b).to(tl.float32)

    mean_sq = tl.sum(xb * xb, axis=0) / N
    inv = 1.0 / tl.sqrt(mean_sq + eps)

    # (xf * rsqrt).to(bf16) * w   -- w in bf16, mult in bf16
    normed = (xb * inv).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = normed * w

    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_rmsnorm_kernel[(m,)](
            y, self.b1, self.rms2_w, out,
            n, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
