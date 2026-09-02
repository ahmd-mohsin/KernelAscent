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
    stride_x, stride_y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)

    # first RMSNorm
    xf = x.to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / D_
    inv1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    y1 = (xf * inv1).to(tl.float16) * w1  # fp16 multiply, matches reference

    # second RMSNorm
    yf = y1.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / D_
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    y2 = (yf * inv2).to(tl.float16) * w2

    tl.store(Y + row * stride_y + cols, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _double_rmsnorm_kernel[(m,)](
            x, self.rms1_w, self.rms2_w, out,
            x.stride(0), out.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
