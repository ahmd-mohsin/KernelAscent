import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 845
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)

    # relu (fp16, exact)
    x = tl.maximum(x, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1: compute in fp32, round back to fp16 (matches PyTorch opmath behavior)
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # gelu #2
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for fp16 softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_act_softmax_kernel[(m,)](
            x, y, n, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
