import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 199
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X, W1, W2, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # First RMSNorm
    ms1 = tl.sum(xf * xf, axis=0) / N
    inv1 = 1.0 / tl.sqrt(ms1 + eps)
    xn1 = (xf * inv1).to(tl.bfloat16)  # cast to bf16 before weight mul (match ref)

    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    x1 = (xn1.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # Second RMSNorm
    x1f = x1.to(tl.float32)
    ms2 = tl.sum(x1f * x1f, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + eps)
    xn2 = (x1f * inv2).to(tl.bfloat16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    y = (xn2.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _double_rmsnorm_kernel[(Mrows,)](
            x, self.rms1_w, self.rms2_w, y,
            x.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
