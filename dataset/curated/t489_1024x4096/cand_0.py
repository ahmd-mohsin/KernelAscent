import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 489
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _ln_kernel(X, G, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


@triton.jit
def _epilogue_kernel(Y, B, OUT, total, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    y = tl.load(Y + offs, mask=mask, other=0.0)
    b = tl.load(B + (offs % N), mask=mask, other=0.0)
    y = tl.maximum(y, 0.0)
    y = y + b
    y = tl.maximum(y, 0.0)
    tl.store(OUT + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        m, n = x.shape
        x = x.contiguous()
        xn = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _ln_kernel[(m,)](x, self.ln0_g, self.ln0_b, xn, n, 1e-5,
                         BLOCK=BLOCK, num_warps=8)

        y = xn @ self.W1  # cuBLAS fp16 tensor-core GEMM

        out = torch.empty_like(y)
        total = y.numel()
        N2 = y.shape[-1]
        BLK = 1024
        grid = (triton.cdiv(total, BLK),)
        _epilogue_kernel[grid](y, self.b3, out, total, N2, BLOCK=BLK, num_warps=4)
        return out
