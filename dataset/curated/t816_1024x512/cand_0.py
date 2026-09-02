import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 816
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _scale_gelu_softmax_kernel(
    X_ptr, Out_ptr,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * SCALE
    # exact GELU: 0.5 * y * (1 + erf(y / sqrt(2)))
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = tl.where(mask, g, float('-inf'))

    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * stride_o + cols, out.to(Out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _scale_gelu_softmax_kernel[(m,)](
            h, out,
            n, h.stride(0), out.stride(0),
            SCALE=1.482,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
