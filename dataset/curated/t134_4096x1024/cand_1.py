import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 134
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _gelu_gelu_softmax_kernel(
    X, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # first GELU (exact, erf-based), computed in fp32 then cast back to fp16
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g1 = g1.to(tl.float16)

    # second GELU
    g1f = g1.to(tl.float32)
    g2 = g1f * 0.5 * (1.0 + tl.math.erf(g1f * INV_SQRT2))
    g2 = g2.to(tl.float16)

    # softmax in fp32
    v = tl.where(mask, g2.to(tl.float32), float('-inf'))
    vmax = tl.max(v, axis=0)
    e = tl.exp(v - vmax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

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
        _gelu_gelu_softmax_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
