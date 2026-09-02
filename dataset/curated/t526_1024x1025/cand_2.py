import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 526
M, D, DT = 1024, 1025, torch.bfloat16


@triton.jit
def _fused_softmax_rms_rms_ln_ln(
    X, Y,
    W1, W2, G3, B3, G4, B4,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    # ---------------- softmax (fp32 accumulate, bf16 output) ----------------
    x = tl.load(X + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, 0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, 0)
    xb = (e / denom).to(tl.bfloat16)

    # ---------------- RMSNorm 1 ----------------
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    xb = (xf * r).to(tl.bfloat16) * w1  # bf16 multiply, matches PyTorch

    # ---------------- RMSNorm 2 ----------------
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    xb = (xf * r).to(tl.bfloat16) * w2

    # ---------------- LayerNorm 3 (fp32 math, affine in fp32) ----------------
    xf = xb.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), 0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (diff * rstd * g3 + b3).to(tl.bfloat16)

    # ---------------- LayerNorm 4 ----------------
    xf = xb.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), 0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd * g4 + b4).to(tl.bfloat16)

    tl.store(Y + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_rms_rms_ln_ln[(rows,)](
            x2, y,
            self.rms1_w, self.rms2_w,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
