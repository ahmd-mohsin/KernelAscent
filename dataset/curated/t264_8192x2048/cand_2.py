import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 264
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_norms_gelu_kernel(
    X, W1, G2, B2, G3, B3, Y,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row * N

    # ---- load matmul output (fp16) ----
    x16 = tl.load(X + base + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # ---- RMSNorm (fp32 stats, cast to fp16, fp16 mul with weight) ----
    ms = tl.sum(xf * xf, axis=0) / N
    rms = tl.math.rsqrt(ms + 1e-6)
    y16 = (xf * rms).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    y16 = y16 * w1  # fp16 arithmetic to match eager

    # ---- LayerNorm 2 (fp32 opmath, fp16 output) ----
    xf = y16.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y16 = (diff * rstd * g2 + b2).to(tl.float16)

    # ---- LayerNorm 3 (fp32 opmath, fp16 output) ----
    xf = y16.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y16 = (diff * rstd * g3 + b3).to(tl.float16)

    # ---- GELU (erf-based, fp32 opmath, fp16 output) ----
    xf = y16.to(tl.float32)
    out = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))

    tl.store(Y + base + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0
        orig_shape = h.shape
        N = orig_shape[-1]
        h = h.contiguous().view(-1, N)
        rows = h.shape[0]
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_norms_gelu_kernel[(rows,)](
            h, self.rms1_w, self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b, out,
            N=N, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
