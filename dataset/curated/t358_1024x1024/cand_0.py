import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 358
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _rmsnorm_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    # match reference: normalize in fp32, cast to bf16, then bf16 multiply by weight
    xn = (xf * rstd).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul then in-place bias add (numerically identical to x@W0 + b1)
        h = torch.matmul(x, self.W0)
        h.add_(self.b1)
        h = torch.matmul(h, self.W2)

        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _rmsnorm_kernel[(m,)](
            h, self.rms3_w, out,
            h.stride(0), out.stride(0),
            n, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
