import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 373
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_ln_gelu(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based); match PyTorch: fp32 opmath then cast back to fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    h = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    h = h.to(tl.float16).to(tl.float32)

    # LayerNorm in fp32 (PyTorch accumulates half LN in float)
    mean = tl.sum(h, axis=0) / N
    d = tl.where(mask, h - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # second GELU
    out = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    tl.store(Y_ptr + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_ln_gelu[(Mrows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
