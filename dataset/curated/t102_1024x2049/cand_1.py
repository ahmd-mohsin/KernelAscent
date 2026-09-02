import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 102
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _fused_bias_double_softmax_scale(
    Y_ptr, B_ptr, O_ptr,
    N, stride_row,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # bias add (fp32 accumulate, round to fp16 to match reference fp16 add)
    x = (y.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # first softmax (fp32 math, fp16 output — matches PyTorch half softmax)
    xf = tl.where(mask, x.to(tl.float32), float('-inf'))
    m1 = tl.max(xf, axis=0)
    e1 = tl.exp(xf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16)

    # second softmax
    pf = tl.where(mask, p1.to(tl.float32), float('-inf'))
    m2 = tl.max(pf, axis=0)
    e2 = tl.exp(pf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.float16)

    # two sequential scalar multiplies (fp32 opmath, round to fp16 each time)
    r = (p2.to(tl.float32) * S1).to(tl.float16)
    r = (r.to(tl.float32) * S2).to(tl.float16)

    tl.store(O_ptr + row * stride_row + cols, r, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM with fp32 accumulate (same as reference)
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_double_softmax_scale[(rows,)](
            y, self.b1, out,
            N, y.stride(0),
            1.3677, 1.255,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
