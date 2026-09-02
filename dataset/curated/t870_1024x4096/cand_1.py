import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 870
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _scale_gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # match reference: scale computed in fp32, rounded to bf16, then gelu in fp32
    x = (x * 1.2973).to(tl.bfloat16).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _bias_softmax_kernel(X, B, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    # match reference: bias add rounded to bf16 before softmax
    v = (x + b).to(tl.bfloat16).to(tl.float32)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W2 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n = x.numel()
        h = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _scale_gelu_kernel[grid](x, h, n, BLOCK=BLOCK, num_warps=4)

        z = h @ self.W2  # cuBLAS bf16 matmul (tensor cores)

        m, d = z.shape
        out = torch.empty_like(z)
        BLOCK_C = triton.next_power_of_2(d)
        _bias_softmax_kernel[(m,)](
            z, self.b3, out, d, z.stride(0), out.stride(0),
            BLOCK=BLOCK_C, num_warps=16,
        )
        return out
