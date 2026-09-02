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
    N, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0
        z = z.contiguous()
        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 1024 else 4
        _scale_softmax_kernel[(m,)](
            z, out,
            z.stride(0), out.stride(0),
            n, 1.1872,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
