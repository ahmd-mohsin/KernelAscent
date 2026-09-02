import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 896
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_relu_bias_softmax(
    X, B, Y,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # relu in bf16, add in bf16 (matches PyTorch elementwise semantics)
    x = tl.maximum(x, 0.0)
    x = x + b

    # softmax with fp32 accumulation (matches PyTorch softmax for bf16)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float("-inf"))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _fused_relu_bias_softmax[(m,)](
            x2, self.b1, y,
            n,
            x2.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
