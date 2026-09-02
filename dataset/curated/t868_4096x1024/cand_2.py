import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 868
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _relu_scale_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # relu then scale
    x = tl.maximum(x, 0.0) * SCALE
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        if not h.is_cuda:
            h = torch.relu(h) * 1.3025
            return torch.softmax(h, dim=-1)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 1024 else 4
        _relu_scale_softmax_kernel[(Mrows,)](
            h, y,
            h.stride(0), y.stride(0),
            N, 1.3025,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
