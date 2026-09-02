import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 588
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X, W0, W1, Y,
    D_: tl.constexpr,
    stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # first RMSNorm
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + eps)
    y = (xf * inv).to(tl.bfloat16)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = y * w0  # bf16 multiply, matches reference

    # second RMSNorm
    yf = y.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / D_
    inv2 = 1.0 / tl.sqrt(ms2 + eps)
    y2 = (yf * inv2).to(tl.bfloat16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y2 = y2 * w1

    tl.store(Y + row * stride_y + cols, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.dtype == torch.bfloat16
        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _double_rmsnorm_kernel[(Mrows,)](
            x, self.rms0_w, self.rms1_w, y,
            Dcols,
            x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
