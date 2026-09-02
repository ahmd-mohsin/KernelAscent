import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 624
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_act_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # exact gelu (erf-based), computed in fp32 then rounded to bf16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    # relu (no-op numerically for gelu(relu(x)) but kept for exactness)
    g = tl.maximum(g, 0.0)

    # RMSNorm in fp32
    ms = tl.sum(g * g, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y = g * r
    # cast to bf16 (matches `.to(x.dtype)` in reference), then multiply by weight in fp32 opmath
    y = y.to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)
    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (same op as reference)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_act_rmsnorm_kernel[(Mrows,)](
            h, self.rms4_w, out,
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
