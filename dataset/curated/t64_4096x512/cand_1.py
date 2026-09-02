import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 64
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X, W1, W2, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # First RMSNorm
    ms1 = tl.sum(xf * xf, axis=0) / N
    inv1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    y1 = (xf * inv1).to(tl.bfloat16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    y1 = y1 * w1  # bf16 multiply, matches reference

    # Second RMSNorm
    xf2 = y1.to(tl.float32)
    ms2 = tl.sum(xf2 * xf2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    y2 = (xf2 * inv2).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    y2 = y2 * w2

    tl.store(Y + row * stride_y + cols, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _double_rmsnorm_kernel[(m,)](
            h, self.rms1_w, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
