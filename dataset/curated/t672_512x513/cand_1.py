import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 672
M, D, DT = 512, 513, torch.float16


@triton.jit
def _relu_bias_kernel(X, B, Y, n_elements, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0)
    b = tl.load(B + (offs % N), mask=mask, other=0.0)
    x = tl.maximum(x, 0.0) + b
    tl.store(Y + offs, x, mask=mask)


@triton.jit
def _layernorm_kernel(X, G, Bias, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Bias + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
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
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0

        # Fused ReLU + bias add (single elementwise Triton kernel)
        h = h.contiguous()
        n = h.numel()
        N1 = h.shape[-1]
        h2 = torch.empty_like(h)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _relu_bias_kernel[grid](h, self.b2, h2, n, N1, BLOCK=BLOCK)

        # GEMM 2 (cuBLAS tensor cores)
        y = h2 @ self.W3

        # Fused LayerNorm in Triton (fp32 accumulation, matches F.layer_norm)
        y = y.contiguous()
        N2 = y.shape[-1]
        rows = y.numel() // N2
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N2)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _layernorm_kernel[(rows,)](
            y, self.ln4_g, self.ln4_b, out, N2, 1e-5,
            BLOCK=BLOCK_N, num_warps=num_warps,
        )
        return out
