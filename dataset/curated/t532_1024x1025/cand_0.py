import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 532
M, D, DT = 1024, 1025, torch.bfloat16


@triton.jit
def _ln_fwd(X, G, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _relu_scale(X, Y, numel, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x, 0.0) * scale
    tl.store(Y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x = x.contiguous().view(-1, N)
        Mrows = x.shape[0]

        # Fused LayerNorm (fp32 accumulation, matches PyTorch CUDA kernel)
        x_ln = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _ln_fwd[(Mrows,)](
            x, self.ln0_g, self.ln0_b, x_ln,
            N, 1e-5, BLOCK=BLOCK_N, num_warps=num_warps,
        )

        # Matmul via cuBLAS (same path as reference)
        h = torch.matmul(x_ln, self.W1)

        # Fused ReLU + scale
        out = torch.empty_like(h)
        numel = h.numel()
        BLOCK = 1024
        grid = (triton.cdiv(numel, BLOCK),)
        _relu_scale[grid](h, out, numel, 1.2948, BLOCK=BLOCK)

        return out.view(*orig_shape[:-1], self.W1.shape[1])
