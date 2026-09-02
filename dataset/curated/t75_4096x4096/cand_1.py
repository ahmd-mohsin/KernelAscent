import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 75
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_epilogue(
    X, RW, G3, B3, G4, B4, Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * N + offs

    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (fp32 math, cast to bf16, then bf16-style weight mul) ----
    ms = tl.sum(x * x, axis=0) / N
    y = x * tl.math.rsqrt(ms + 1e-6)
    y = y.to(tl.bfloat16)  # match .to(x.dtype)

    w = tl.load(RW + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w).to(tl.bfloat16)  # bf16 * bf16 -> bf16 (single rounding)

    # ---- GELU (erf-based, fp32 opmath as PyTorch does for bf16) ----
    t = y.to(tl.float32)
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # ---- LayerNorm 3 (fp32 accumulate, bf16 out) ----
    t = g.to(tl.float32)
    mean = tl.sum(tl.where(mask, t, 0.0), axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    o = ((t - mean) * rstd * g3 + b3).to(tl.bfloat16)

    # ---- LayerNorm 4 (on bf16 output of LN3) ----
    t = o.to(tl.float32)
    mean = tl.sum(tl.where(mask, t, 0.0), axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    o = ((t - mean) * rstd * g4 + b4).to(tl.bfloat16)

    tl.store(Out + row * N + offs, o, mask=mask)


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
        # GEMM via cuBLAS tensor cores (bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.view(-1, N)
        rows = h2.shape[0]

        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue[(rows,)](
            h2, self.rms1_w, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
