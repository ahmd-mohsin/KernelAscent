import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 889
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _scale_gelu_softmax_kernel(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N, SCALE,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # scale
    x = x * SCALE

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # softmax over the row
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        if not h.is_cuda:
            h = h * 1.1778
            h = F.gelu(h)
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _scale_gelu_softmax_kernel[(Mrows,)](
            h, y,
            h.stride(0), y.stride(0),
            N, 1.1778,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
