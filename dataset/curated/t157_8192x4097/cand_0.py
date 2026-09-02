import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 157
M, D, DT = 8192, 4097, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    N,
    stride_x, stride_y,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # Emulate PyTorch half elementwise mul: compute in fp32, round to fp16 each step
    xf = x.to(tl.float32)
    xf = (xf * S1).to(tl.float16).to(tl.float32)
    xf = (xf * S2).to(tl.float16).to(tl.float32)

    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM (fp16)
        z = x @ self.W0

        Mrows, N = z.shape
        y = torch.empty_like(z)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _scale_softmax_kernel[(Mrows,)](
            z, y,
            N,
            z.stride(0), y.stride(0),
            1.0453, 1.0092,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
