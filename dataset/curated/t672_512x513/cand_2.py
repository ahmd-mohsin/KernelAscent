import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 672
M, D, DT = 512, 513, torch.float16


@triton.jit
def _relu_bias_kernel(X, B, n_elem, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(X + offs, mask=mask, other=0.0)
    b = tl.load(B + (offs % N), mask=mask, other=0.0)
    zero = x - x  # zero of same dtype
    y = tl.where(x > 0, x, zero) + b
    tl.store(X + offs, y, mask=mask)


@triton.jit
def _layernorm_kernel(X, G, Bp, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Bp + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            h = x @ self.W0
            h = torch.relu(h) + self.b2
            h = h @ self.W3
            return F.layer_norm(h, (h.shape[-1],), self.ln4_g, self.ln4_b)

        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        n_elem = h.numel()
        N1 = h.shape[-1]
        BLOCK = 1024
        _relu_bias_kernel[(triton.cdiv(n_elem, BLOCK),)](
            h, self.b2, n_elem, N1, BLOCK=BLOCK
        )

        # GEMM 2 (cuBLAS tensor cores)
        h2 = h @ self.W3
        h2 = h2.contiguous()
        rows = h2.numel() // h2.shape[-1]
        N2 = h2.shape[-1]
        y = torch.empty_like(h2)
        BLOCK_N = triton.next_power_of_2(N2)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _layernorm_kernel[(rows,)](
            h2, self.ln4_g, self.ln4_b, y, N2, 1e-5,
            BLOCK=BLOCK_N, num_warps=num_warps
        )
        return y
