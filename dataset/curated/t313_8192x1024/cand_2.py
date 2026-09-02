import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 313
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _relu_ln_kernel(X, G, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gelu_scale_kernel(X, Y, n_elements, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))), computed in fp32
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # match reference rounding: gelu -> bf16, then scale in fp32 -> bf16
    g = g.to(tl.bfloat16).to(tl.float32)
    y = (g * scale).to(tl.bfloat16)
    tl.store(Y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(N)
        _relu_ln_kernel[(Mrows,)](
            x, self.ln1_g, self.ln1_b, y, N, 1e-5,
            BLOCK=BLOCK, num_warps=8,
        )

        out = y @ self.W2  # cuBLAS bf16 matmul (tensor cores)

        n = out.numel()
        BLOCK2 = 1024
        grid = (triton.cdiv(n, BLOCK2),)
        _gelu_scale_kernel[grid](out, out, n, 1.1733, BLOCK=BLOCK2, num_warps=4)
        return out
