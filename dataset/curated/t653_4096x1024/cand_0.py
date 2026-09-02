import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 653
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, W0, W1, B, Y, N, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # --- RMSNorm 0 ---
    r = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w0).to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm 1 ---
    r = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w1).to(tl.bfloat16).to(tl.float32)

    # --- GELU (exact, erf) twice, rounding to bf16 between ops like eager ---
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # --- bias add ---
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b

    tl.store(Y + row * N + cols, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # fallback: reference path
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.gelu(x)
            x = F.gelu(x)
            x = x + self.b4
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, self.rms0_w, self.rms1_w, self.b4, y,
            N, 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
