import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 773
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_double_softmax_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 compute, round to bf16 like PyTorch output)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    x1 = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # softmax 2
    x1m = tl.where(mask, x1, float('-inf'))
    m2 = tl.max(x1m, axis=0)
    e2 = tl.exp(x1m - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    x2 = (e2 / s2).to(tl.bfloat16).to(tl.float32)

    # relu
    x2 = tl.maximum(x2, 0.0)

    # + bias (bf16 rounding after add, matching bf16 tensor add)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x3 = (x2 + b).to(tl.bfloat16).to(tl.float32)

    # * 1.3071 (bf16 rounding after mul)
    y = (x3 * 1.3071).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        b = self.b3
        if b.device != x.device:
            b = b.to(x.device)
        BLOCK = triton.next_power_of_2(d)
        _fused_double_softmax_kernel[(m,)](
            x, b, y,
            x.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
