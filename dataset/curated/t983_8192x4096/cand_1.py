import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 983
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _rmsnorm_relu_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    xn = (x * rstd).to(tl.bfloat16)  # match .to(x.dtype)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = xn.to(tl.float32) * w        # bf16*bf16 computed in fp32, rounded on store
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_ym + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _rmsnorm_relu_kernel[(Mrows,)](
            x, self.rms0_w, y,
            x.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
