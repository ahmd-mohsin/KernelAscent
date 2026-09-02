import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 657
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, W0, W1, Y, N, stride_x, stride_y, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm 0
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    # cast normalized value to bf16, then multiply by bf16 weight (bf16 mul)
    x1 = (xf * r).to(tl.bfloat16) * w0

    # RMSNorm 1
    x1f = x1.to(tl.float32)
    ms1 = tl.sum(tl.where(mask, x1f * x1f, 0.0), axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + eps)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    x2 = (x1f * r1).to(tl.bfloat16) * w1

    # GELU (exact, erf) computed in fp32, output cast to bf16 each time
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    v = x2.to(tl.float32)
    g1 = (0.5 * v * (1.0 + tl.math.erf(v * INV_SQRT2))).to(tl.bfloat16)
    v2 = g1.to(tl.float32)
    g2 = (0.5 * v2 * (1.0 + tl.math.erf(v2 * INV_SQRT2))).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, g2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda or True
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x, self.rms0_w, self.rms1_w, y,
            N, x.stride(0), y.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
