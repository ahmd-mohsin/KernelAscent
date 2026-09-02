import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 23
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _gelu_bias_kernel(X, B, N_COLS, n_elem, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU in fp32, then round to bf16 (matches PyTorch op semantics)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    b = tl.load(B + (offs % N_COLS), mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16)
    tl.store(X + offs, y, mask=mask)


@triton.jit
def _bias_bias_softmax_kernel(X, B4, B5, Out, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + offs, mask=mask, other=0.0).to(tl.float32)
    # replicate the two separate bf16 add ops (each rounds to bf16)
    x = (x + b4).to(tl.bfloat16).to(tl.float32)
    x = (x + b5).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.math.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16)
    tl.store(Out + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        n_elem = h.numel()
        n_cols = h.shape[-1]
        BLOCK = 1024
        grid = (triton.cdiv(n_elem, BLOCK),)
        _gelu_bias_kernel[grid](h, self.b2, n_cols, n_elem, BLOCK=BLOCK, num_warps=4)

        # GEMM 2 (cuBLAS tensor cores)
        y = h @ self.W3
        y = y.contiguous()
        rows = y.numel() // y.shape[-1]
        N = y.shape[-1]
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _bias_bias_softmax_kernel[(rows,)](y, self.b4, self.b5, out, N, BLOCK=BLOCK_N, num_warps=4)
        return out
