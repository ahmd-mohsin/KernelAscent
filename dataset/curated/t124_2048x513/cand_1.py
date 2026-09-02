import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 124
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_relu_bias_softmax_scale(
    X, B, Y,
    stride_xm, stride_ym,
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # relu (applied twice == once), then add bias; do the add in fp16 to match
    # reference elementwise semantics, then upcast for softmax accumulation.
    x = tl.maximum(x, 0.0)
    x = x + b  # fp16 add, matches reference
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float("-inf"))

    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s) * scale

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_bias_softmax_scale[(m,)](
            h, self.b3, y,
            h.stride(0), y.stride(0),
            n, 1.1092,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
