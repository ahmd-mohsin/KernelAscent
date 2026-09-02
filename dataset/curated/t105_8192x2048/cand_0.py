import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 105
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X, W1, W2, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First RMSNorm
    ms1 = tl.sum(x * x, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(ms1 + eps)
    x16 = (x * rstd1).to(tl.float16)  # round to fp16 like .to(x.dtype)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    y16 = (x16.to(tl.float32) * w1).to(tl.float16)  # fp16 elementwise mul

    # Second RMSNorm
    y = y16.to(tl.float32)
    ms2 = tl.sum(y * y, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(ms2 + eps)
    y16b = (y * rstd2).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y16b.to(tl.float32) * w2).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _double_rmsnorm_kernel[(M_,)](
            x, self.rms1_w, self.rms2_w, y,
            N_, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
