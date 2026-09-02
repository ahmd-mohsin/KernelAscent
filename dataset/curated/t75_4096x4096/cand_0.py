import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 75
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_rms_gelu_ln_ln(
    X, OUT, W_RMS, G3, B3, G4, B4,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (fp32 compute, bf16 round-trips to match reference) ----
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_RMS + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact erf, fp32 opmath like PyTorch bf16 kernel) ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd * g3 + b3).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 4 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g4 + b4).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores (already optimal on A100)
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_gelu_ln_ln[(m,)](
            x, out,
            self.rms1_w, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b,
            n, x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
