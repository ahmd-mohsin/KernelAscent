import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU, computed in fp32 like PyTorch's CUDA kernel
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _scale_softmax_kernel(X, Y, n_cols, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + row * n_cols + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * n_cols + offs, y.to(tl.bfloat16), mask=mask)


SEED = 863
M, D, DT = 8192, 2048, torch.bfloat16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x = x.contiguous()
        n = x.numel()

        # Fused GELU (elementwise, fp32 internal math, bf16 output)
        g = torch.empty_like(x)
        BLOCK = 4096
        grid = (triton.cdiv(n, BLOCK),)
        _gelu_kernel[grid](x, g, n, BLOCK=BLOCK, num_warps=8)

        # Matmul via cuBLAS tensor cores
        h = g.view(-1, orig_shape[-1]) @ self.W1
        h = h.contiguous()

        # Fused scale + softmax (one row per program, fp32 internal math)
        rows, cols = h.shape
        out = torch.empty_like(h)
        BLOCK_C = triton.next_power_of_2(cols)
        _scale_softmax_kernel[(rows,)](h, out, cols, 1.0993, BLOCK=BLOCK_C, num_warps=8)

        return out.view(*orig_shape[:-1], self.W1.shape[1])
