import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 34
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_bias_relu_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add + relu in fp16 (matches reference elementwise ops)
    v = x + b
    zero = v - v
    v = tl.where(v > zero, v, zero)

    # softmax accumulated in fp32 (matches PyTorch CUDA softmax accumulation)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, -float('inf'))
    row_max = tl.max(vf, axis=0)
    vf = vf - row_max
    num = tl.exp(vf)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_relu_softmax[(m,)](
            x, self.b0, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
