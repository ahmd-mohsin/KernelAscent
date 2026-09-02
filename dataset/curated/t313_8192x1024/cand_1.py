import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 313
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _relu_ln_kernel(X, G, B, Y, N, eps,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gelu_scale_kernel(X, Y, n_elem, scale,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf-based) GELU in fp32, round to bf16 (matches F.gelu on bf16)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16).to(tl.float32)
    # then scale in fp32 opmath, round to bf16 (matches bf16 * python-float)
    out = g_bf * scale
    tl.store(Y + offs, out.to(tl.bfloat16), mask=mask)


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
        z = y @ self.W2
        out = torch.empty_like(z)
        n_elem = z.numel()
        BLK = 1024
        grid = (triton.cdiv(n_elem, BLK),)
        _gelu_scale_kernel[grid](
            z, out, n_elem, 1.1733,
            BLOCK=BLK, num_warps=4,
        )
        return out
