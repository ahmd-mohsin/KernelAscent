import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 906
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_relu_softmax_kernel(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # exact GELU (erf-based), computed in fp32 like PyTorch's opmath
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    # cast back to bf16 to mirror PyTorch's per-op output dtype
    g = g.to(X_ptr.dtype.element_ty)

    # ReLU
    r = tl.maximum(g, 0.0)

    # softmax in fp32 (matches PyTorch internal accumulation)
    rf = r.to(tl.float32)
    rf = tl.where(mask, rf, float('-inf'))
    row_max = tl.max(rf, axis=0)
    e = tl.exp(rf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * stride_ym + cols, out.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
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
        _fused_gelu_relu_softmax_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
