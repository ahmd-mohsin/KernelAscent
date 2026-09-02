import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 863
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _scale_softmax_kernel(X, Y, n_cols, stride_x, stride_y,
                          SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # match reference: (x * scale) rounded to bf16, then softmax in fp32
    xs = (x * SCALE).to(tl.bfloat16).to(tl.float32)
    xs = tl.where(mask, xs, float('-inf'))
    m = tl.max(xs, axis=0)
    e = tl.exp(xs - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n = x.numel()
        g = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gelu_kernel[grid](x, g, n, BLOCK=BLOCK, num_warps=4)

        z = g @ self.W1  # cuBLAS bf16 matmul (tensor cores)

        rows, cols = z.shape
        out = torch.empty_like(z)
        BLOCK_C = triton.next_power_of_2(cols)
        _scale_softmax_kernel[(rows,)](
            z, out, cols, z.stride(0), out.stride(0),
            SCALE=1.0993, BLOCK=BLOCK_C, num_warps=8,
        )
        return out
