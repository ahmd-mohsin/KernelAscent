import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 298
M, D, DT = 1024, 1024, torch.bfloat16

INV_SQRT2 = 0.7071067811865476


@triton.jit
def _softmax_gelu_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    denom = tl.sum(e, 0)
    s = e / denom
    # match reference: softmax result is rounded to input dtype before relu/gelu
    s = s.to(X.dtype.element_ty).to(tl.float32)
    # relu is a no-op on softmax output (>= 0)
    g = 0.5 * s * (1.0 + tl.math.erf(s * INV_SQRT2))
    tl.store(Y + row * N + offs, g.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_kernel(X, Y, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    tl.store(Y + offs, g.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape

        # Fused softmax + relu + gelu (single pass per row)
        out1 = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _softmax_gelu_kernel[(m,)](h, out1, n, BLOCK=BLOCK, num_warps=16)

        # GEMM 2
        h2 = out1 @ self.W4
        h2 = h2.contiguous()

        # Fused final gelu
        out2 = torch.empty_like(h2)
        numel = h2.numel()
        BLOCK2 = 1024
        grid = (triton.cdiv(numel, BLOCK2),)
        _gelu_kernel[grid](h2, out2, numel, BLOCK=BLOCK2, num_warps=4)
        return out2
