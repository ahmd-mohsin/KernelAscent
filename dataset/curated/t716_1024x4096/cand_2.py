import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 716
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU, computed in fp32 (matches PyTorch opmath), rounded to fp16
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _bias_softmax_kernel(X, B, Y, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    # bias add rounded through fp16 to exactly match `x + b2` in half precision
    t = (x + b).to(tl.float16).to(tl.float32)
    t = tl.where(mask, t, float('-inf'))
    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n = x.numel()
        g = torch.empty_like(x)

        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gelu_kernel[grid](x, g, n, BLOCK=BLOCK, num_warps=4)

        # cuBLAS fp16 GEMM with fp32 accumulate (tensor cores on A100)
        y = g @ self.W1

        m, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _bias_softmax_kernel[(m,)](y, self.b2, out, N, BLOCK_N=BLOCK_N, num_warps=8)
        return out
