import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 648
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _double_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 accumulate, like PyTorch on half inputs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (intermediate tensor dtype in reference) then back
    p = p.to(tl.float16).to(tl.float32)

    # second softmax
    p = tl.where(mask, p, float('-inf'))
    m2 = tl.max(p, axis=0)
    e2 = tl.exp(p - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores on A100)
        m, n = z.shape
        y = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _double_softmax_kernel[(m,)](
            z, y, n, z.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        return y
