import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 182
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, ncols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < ncols
    x = tl.load(X + row * stride + cols, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_bias_kernel(X, B, Y, n_elements, NCOLS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU (erf-based), computed in fp32 like PyTorch's opmath for bf16
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round to bf16 to match the intermediate tensor produced by F.gelu,
    # then upcast for the bias add (matching bf16 + bf16 elementwise semantics)
    g = g.to(tl.bfloat16).to(tl.float32)
    b = tl.load(B + (offs % NCOLS), mask=mask, other=0.0).to(tl.float32)
    y = g + b
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mr, Dr = x.shape

        # Fused row-wise softmax (fp32 accumulation, bf16 output)
        s = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dr)
        _softmax_kernel[(Mr,)](
            x, s, Dr, x.stride(0),
            BLOCK=BLOCK, num_warps=16,
        )

        # cuBLAS bf16 GEMM with tensor cores
        y = s @ self.W1

        # Fused exact-GELU + bias add
        out = torch.empty_like(y)
        n = y.numel()
        BLOCK2 = 1024
        grid = (triton.cdiv(n, BLOCK2),)
        _gelu_bias_kernel[grid](
            y, self.b3, out, n,
            NCOLS=y.shape[1], BLOCK=BLOCK2, num_warps=4,
        )
        return out
