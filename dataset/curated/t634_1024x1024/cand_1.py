import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 634
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax 1 (fp32 accumulate, like PyTorch's half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1

    # cast to fp16 (output dtype of first softmax), scale in fp16
    p1_h = p1.to(tl.float16)
    scale = tl.full((), 1.4236, tl.float16)
    z_h = p1_h * scale
    z = z_h.to(tl.float32)
    z = tl.where(mask, z, -float('inf'))

    # softmax 2
    m2 = tl.max(z, axis=0)
    e2 = tl.exp(z - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    # cast to fp16 then gelu (erf) in fp32, like PyTorch half gelu
    g_in = p2.to(tl.float16).to(tl.float32)
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    g = 0.5 * g_in * (1.0 + tl.math.erf(g_in * inv_sqrt2))
    g_h = g.to(tl.float16)

    # relu
    zero = tl.full((), 0.0, tl.float16)
    out = tl.maximum(g_h, zero)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_kernel[(n_rows,)](
            x2, y, n_cols, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
