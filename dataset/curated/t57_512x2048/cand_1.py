import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 57
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_relu_gelu2_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # relu (in input dtype)
    x = tl.maximum(x, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1: compute in fp32, round back to fp16 (matches per-op PyTorch behavior)
    xf = x.to(tl.float32)
    g1 = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g1 = g1.to(X.dtype.element_ty)

    # gelu #2
    g1f = g1.to(tl.float32)
    g2 = g1f * 0.5 * (1.0 + tl.math.erf(g1f * INV_SQRT2))
    g2 = g2.to(X.dtype.element_ty)

    # softmax in fp32
    z = g2.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _fused_relu_gelu2_softmax_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
