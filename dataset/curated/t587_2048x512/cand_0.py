import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 587
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_norm_gelu_kernel(
    X, LG, LB, W2, W5, OUT,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 internal, bf16 output) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    g = tl.load(LG + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LB + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm #1 (fp32, round to bf16, then bf16*bf16 -> bf16) ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    y = (y * tl.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w2).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact erf) twice, rounded to bf16 between ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = (0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    y = (0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm #2 ----
    ms2 = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    y = (y * tl.rsqrt(ms2 + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w5 = tl.load(W5 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w5).to(tl.bfloat16)

    tl.store(OUT + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        rows, N = y.shape[0], y.shape[1]
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_gelu_kernel[(rows,)](
            y, self.ln1_g, self.ln1_b, self.rms2_w, self.rms5_w, out,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
