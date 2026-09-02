import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 817
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_relu_scale_softmax(X, Y, N, stride_x, stride_y, SCALE: tl.constexpr,
                              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # relu -> scale -> relu  (scale > 0, so equals relu(x) * scale)
    x = tl.maximum(x, 0.0) * SCALE
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x) * 1.0204
            return torch.softmax(torch.relu(x), dim=-1)
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_relu_scale_softmax[(m,)](
            x2, y, n, x2.stride(0), y.stride(0), 1.0204,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
