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
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # GELU (exact, erf-based) applied twice
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # Softmax over the row
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Y_ptr + row * stride_y + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_act_softmax_kernel[(Mrows,)](
            h, out,
            N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
