import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 945
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _softmax_bias_gelu_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's half softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # round to fp16 (matches torch.softmax output dtype), then add bias in fp16
    s16 = s.to(tl.float16)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float16)
    t16 = s16 + b

    # exact gelu computed in fp32 (matches PyTorch opmath), output fp16
    t = t16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * t * (1.0 + tl.math.erf(t * INV_SQRT2))

    tl.store(Y_ptr + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _softmax_bias_gelu_kernel[(m,)](
            h, self.b2, y,
            n, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
