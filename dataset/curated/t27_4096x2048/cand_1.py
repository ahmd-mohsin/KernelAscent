import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 27
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _rmsnorm_scale_kernel(
    X, W, Y,
    N, eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (x * rstd).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xn * w) * scale
    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _rmsnorm_scale_kernel[(m,)](
            x, self.rms1_w, y,
            n, 1e-6, 1.1239,
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        return y
