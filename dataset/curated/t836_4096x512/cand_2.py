import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 836
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _gelu_kernel(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(X + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _softmax_kernel(X, Y, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        h = h.contiguous()
        n = h.numel()
        BLOCK = 1024
        _gelu_kernel[(triton.cdiv(n, BLOCK),)](h, n, BLOCK=BLOCK, num_warps=4)
        z = h @ self.W2
        z = z.contiguous()
        rows, cols = z.shape
        out = torch.empty_like(z)
        _softmax_kernel[(rows,)](z, out, z.stride(0), cols,
                                 BLOCK=triton.next_power_of_2(cols), num_warps=8)
        return out
