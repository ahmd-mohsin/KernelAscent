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
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact (erf-based) GELU in fp32
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * inv_sqrt2))

    # round to bf16 to match PyTorch's elementwise op output dtype, then relu
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.maximum(g, 0.0)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 inputs)
    g = tl.where(mask, g, float("-inf"))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
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
            x = F.gelu(x)
            x = torch.relu(x)
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

        _fused_gelu_relu_softmax_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
