import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 302
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X, W1, W2, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)

    # First RMSNorm
    xf = x.to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + EPS)
    y = (xf * r1).to(tl.float16) * w1  # fp16 multiply, matches reference

    # Second RMSNorm
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + EPS)
    z = (yf * r2).to(tl.float16) * w2

    tl.store(Y + row * stride_y + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _double_rmsnorm_kernel[(rows,)](
            h, self.rms1_w, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N=N, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
