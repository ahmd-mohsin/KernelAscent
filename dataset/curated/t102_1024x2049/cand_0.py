import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 102
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _fused_bias_dsoftmax_scale(
    X, B, Y,
    N, stride_xm, stride_ym,
    S1, S2,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp16 (matches reference fp16 add), then round
    x = (x + b).to(tl.float16)

    # first softmax (fp32 accumulation, output rounded to fp16)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m1 = tl.max(xf, axis=0)
    e1 = tl.exp(xf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = (e1 / s1).to(tl.float16)

    # second softmax
    yf = y1.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m2 = tl.max(yf, axis=0)
    e2 = tl.exp(yf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = (e2 / s2).to(tl.float16)

    # two scalar multiplies, each rounded to fp16 (matches reference)
    y3 = (y2 * S1).to(tl.float16)
    y4 = (y3 * S2).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, y4, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        y = x @ self.W0

        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_bias_dsoftmax_scale[(m,)](
            y, self.b1, out,
            n, y.stride(0), out.stride(0),
            1.3677, 1.255,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
