import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 980
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _relu_scale_softmax_kernel(
    X, Y,
    stride_x, stride_y,
    N, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)
    # relu(relu(x)) * scale
    x = tl.where(x > 0.0, x, 0.0) * SCALE
    x = tl.where(mask, x, -float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        rows, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        if BLOCK >= 8192:
            num_warps = 16
        _relu_scale_softmax_kernel[(rows,)](
            h, y,
            h.stride(0), y.stride(0),
            n, 1.4048,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
