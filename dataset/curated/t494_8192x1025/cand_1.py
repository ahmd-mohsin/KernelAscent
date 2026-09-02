import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 494
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_norms_gelu_kernel(
    X, W1, W2, G, B, Y,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    Nf = N.to(tl.float32)

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 1 (compute in fp32, round to bf16, bf16-style multiply by w1) ----
    r1 = tl.math.rsqrt(tl.sum(x * x, axis=0) / Nf + 1e-6)
    x = (x * r1).to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w1).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 2 ----
    r2 = tl.math.rsqrt(tl.sum(x * x, axis=0) / Nf + 1e-6)
    x = (x * r2).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 internals, bf16 output) ----
    mean = tl.sum(x, axis=0) / Nf
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / Nf
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (erf variant, fp32 math, bf16 output) ----
    y = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * stride + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, self.rms2_w, self.ln3_g, self.ln3_b, y,
            N, x.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
