import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 298
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _softmax_gelu_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * N + offs
    x = tl.load(ptr, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # relu(p) == p since softmax output is non-negative
    g = p * 0.5 * (1.0 + tl.math.erf(p * 0.7071067811865476))
    tl.store(Y + row * N + offs, g.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_kernel(X, Y, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, g.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS bf16 tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape

        # Fused softmax + relu(no-op) + gelu in one Triton kernel
        h2 = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_gelu_kernel[(rows,)](h, h2, N, BLOCK=BLOCK, num_warps=8)

        # GEMM 2
        out = torch.matmul(h2, self.W4)
        out = out.contiguous()

        # Fused elementwise gelu
        result = torch.empty_like(out)
        numel = out.numel()
        BLOCK_E = 1024
        grid = (triton.cdiv(numel, BLOCK_E),)
        _gelu_kernel[grid](out, result, numel, BLOCK=BLOCK_E, num_warps=4)
        return result
