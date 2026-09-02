import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 80
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _softmax_gelu_x2_kernel(
    X_ptr, Y_ptr,
    N,
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 math, round to bf16 like PyTorch output)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu 1 (exact, erf-based)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # softmax 2
    x = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x, axis=0)
    e2 = tl.exp(x - m2)
    s2 = tl.sum(tl.where(mask, e2, 0.0), axis=0)
    x = e2 / s2
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu 2
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    tl.store(Y_ptr + row * stride_ym + cols, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_gelu_x2_kernel[(m,)](
            h, y, n,
            h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
