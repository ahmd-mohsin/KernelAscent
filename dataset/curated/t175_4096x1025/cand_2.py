import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 175
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _relu_double_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # relu in fp16 (matches torch.relu on fp16), then upcast for softmax math
    x = tl.where(x > 0, x, 0.0)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    # first softmax (fp32 math, like PyTorch's half softmax)
    m1 = tl.max(xf, axis=0)
    e1 = tl.exp(xf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = e1 / s1

    # round to fp16 (PyTorch writes intermediate as half), then upcast again
    y1h = y1.to(tl.float16)
    x2 = y1h.to(tl.float32)
    x2 = tl.where(mask, x2, float('-inf'))

    # second softmax
    m2 = tl.max(x2, axis=0)
    e2 = tl.exp(x2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = e2 / s2

    tl.store(Y + row * stride_ym + cols, y2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 matmul
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_double_softmax_kernel[(Mrows,)](
            h, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
