import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 810
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Out_ptr,
    N,  # number of columns
    stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)

    # scale (compute in fp32, round to fp16 like PyTorch half elementwise)
    xf = x.to(tl.float32) * SCALE
    x = xf.to(tl.float16)

    # gelu #1 (exact erf, fp32 opmath, round to fp16)
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = g.to(tl.float16)

    # gelu #2
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = g.to(tl.float16)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax (fp32 accumulation)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores on A100)
        y = x @ self.W0
        y = y.contiguous()

        m, n = y.shape
        out = torch.empty_like(y)

        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _fused_act_softmax_kernel[(m,)](
            y, out,
            n,
            y.stride(0), out.stride(0),
            SCALE=1.2221,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
