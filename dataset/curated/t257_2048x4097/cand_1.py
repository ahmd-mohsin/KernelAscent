import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 257
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    N, stride_x, stride_y,
    scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0
        z = z.contiguous()
        M_, N_ = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(N_)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _scale_softmax_kernel[(M_,)](
            z, out, N_, z.stride(0), out.stride(0),
            1.1872, BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        return out
