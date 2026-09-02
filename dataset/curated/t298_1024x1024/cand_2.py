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
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to bf16 (matches PyTorch storing softmax output in bf16, relu is identity)
    p = p.to(tl.bfloat16).to(tl.float32)
    # erf-based GELU
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
        # matmul 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 4096)
        h = h.contiguous()
        Mrows, N = h.shape

        # fused softmax + relu(identity) + gelu
        h2 = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_gelu_kernel[(Mrows,)](h, h2, N, BLOCK=BLOCK, num_warps=8)

        # matmul 2
        out = h2 @ self.W4
        out = out.contiguous()

        # fused gelu
        y = torch.empty_like(out)
        numel = out.numel()
        BLOCK2 = 1024
        grid = (triton.cdiv(numel, BLOCK2),)
        _gelu_kernel[grid](out, y, numel, BLOCK=BLOCK2, num_warps=4)
        return y
