import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 756
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _rmsnorm_scale_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, eps, alpha,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (x * rstd).to(tl.bfloat16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = (xn * w) * alpha
    tl.store(Out_ptr + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_scale_kernel[(Mrows,)](
            x, self.rms1_w, out,
            N, 1e-6, 1.1814,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
