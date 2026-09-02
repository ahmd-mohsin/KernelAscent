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
    # replicate: (x * 1.2973) computed in fp32, rounded to bf16
    x = (x * 1.2973).to(tl.bfloat16).to(tl.float32)
    # exact (erf) GELU in fp32
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _bias_softmax_kernel(X, B, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    # replicate: bias add in fp32, rounded to bf16, then softmax in fp32
    x = (x + b).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * N + offs, out.to(tl.bfloat16), mask=mask)


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

        # fused scale + exact GELU
        h = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _scale_gelu_kernel[grid](x, h, n, BLOCK=BLOCK, num_warps=4)

        # matmul via cuBLAS tensor cores
        z = h @ self.W2

        # fused bias add + softmax (one row per program)
        out = torch.empty_like(z)
        N = z.shape[1]
        BLOCK_N = triton.next_power_of_2(N)
        _bias_softmax_kernel[(z.shape[0],)](z, self.b3, out, N, BLOCK=BLOCK_N, num_warps=8)

        return out
