import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 870
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _scale_gelu_kernel(X, Y, n_elements, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * SCALE
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bias_softmax_kernel(X, B, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    # add in bf16 (rounded) to match x + b3 semantics, then softmax in fp32
    z = (x + b).to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(Y + row * stride_y + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W2 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        n = x.numel()
        h = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _scale_gelu_kernel[grid](x, h, n, 1.2973, BLOCK=BLOCK, num_warps=4)

        y = h @ self.W2

        out = torch.empty_like(y)
        n_cols = y.shape[1]
        BLOCK_C = triton.next_power_of_2(n_cols)
        _bias_softmax_kernel[(y.shape[0],)](
            y, self.b3, out, n_cols, y.stride(0), out.stride(0),
            BLOCK=BLOCK_C, num_warps=16,
        )
        return out
