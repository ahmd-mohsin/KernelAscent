import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 565
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_norm_softmax_gelu_kernel(
    X, Y,
    LN1G, LN1B, RMSW, LN5G, LN5B,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ----- LayerNorm 1 (fp32 compute, bf16 output) -----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(LN1G + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(LN1B + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (xc * rstd * g1 + b1).to(tl.bfloat16)

    # ----- RMSNorm (fp32 compute, cast to bf16, then bf16 multiply by weight) -----
    xf = xb.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    rr = 1.0 / tl.sqrt(ms + 1e-6)
    xb = (xf * rr).to(tl.bfloat16)
    w = tl.load(RMSW + offs, mask=mask, other=0.0)
    xb = xb * w  # bf16 * bf16

    # ----- Softmax (fp32 compute, bf16 output) -----
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, float("-inf"))
    mmax = tl.max(xf, axis=0)
    e = tl.exp(xf - mmax)
    e = tl.where(mask, e, 0.0)
    ssum = tl.sum(e, axis=0)
    xb = (e / ssum).to(tl.bfloat16)

    # ----- GELU exact (erf), fp32 compute, bf16 output -----
    xf = xb.to(tl.float32)
    gelu = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    xb = gelu.to(tl.bfloat16)

    # ----- LayerNorm 5 (fp32 compute, bf16 output) -----
    xf = xb.to(tl.float32)
    mean2 = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    xc2 = tl.where(mask, xf - mean2, 0.0)
    var2 = tl.sum(xc2 * xc2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g5 = tl.load(LN5G + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(LN5B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xc2 * rstd2 * g5 + b5).to(tl.bfloat16)

    tl.store(Y + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_norm_softmax_gelu_kernel[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.rms2_w, self.ln5_g, self.ln5_b,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=4,
        )
        return out
