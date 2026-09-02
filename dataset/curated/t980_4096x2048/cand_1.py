import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 980
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _relu_scale_softmax_kernel(
    X, Y,
    N,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)  # fp16

    # relu (applying it twice is identical to once) in fp16
    x = tl.maximum(x, x * 0)  # relu, stays fp16
    # multiply by scale in fp16 (match reference half-precision arithmetic)
    s = tl.full((1,), SCALE, dtype=tl.float16)
    x = x * s  # fp16

    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    xf = xf - row_max
    num = tl.exp(xf)
    num = tl.where(mask, num, 0.0)
    den = tl.sum(num, axis=0)
    out = num / den

    tl.store(Y + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        z = x @ self.W0
        z = z.contiguous()
        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_scale_softmax_kernel[(m,)](
            z, out,
            n,
            z.stride(0), out.stride(0),
            SCALE=1.4048,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
