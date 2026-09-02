import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 188
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_relu_bias_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # relu + bias add in fp16 (matches reference dtype semantics)
    t = tl.maximum(x, 0.0)
    t = t + b

    # softmax in fp32 (matches PyTorch's fp32 accumulation for half inputs)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    m = tl.max(tf, axis=0)
    e = tl.exp(tf - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_bias_softmax[(m,)](
            x, self.b1, y,
            x.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
