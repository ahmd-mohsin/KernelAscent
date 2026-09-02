import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 814
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _double_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 math, like PyTorch's half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = e1 / s1

    # round to fp16 (reference materializes fp16 between softmaxes)
    y1 = y1.to(tl.float16).to(tl.float32)
    y1 = tl.where(mask, y1, float('-inf'))

    # second softmax
    m2 = tl.max(y1, axis=0)
    e2 = tl.exp(y1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = e2 / s2

    tl.store(Y + row * stride_ym + cols, y2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _double_softmax_kernel[(m,)](
            x, y,
            x.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
