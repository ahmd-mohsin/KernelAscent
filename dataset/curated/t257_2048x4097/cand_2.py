import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 257
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # scale in fp16 to match reference (x * 1.1872 done on half tensor)
    scale_h = tl.full((), SCALE, dtype=tl.float16)
    x = (x * scale_h).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0
        z = z.contiguous()
        m, n = z.shape
        y = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _scale_softmax_kernel[(m,)](
            z, y,
            z.stride(0), y.stride(0),
            n,
            SCALE=1.1872,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
