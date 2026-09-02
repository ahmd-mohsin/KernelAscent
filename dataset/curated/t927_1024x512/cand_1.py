import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 927
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, OUT,
    G1, B1, G2, B2, RW,
    N, stride_x, stride_o,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    nf = N.to(tl.float32)

    # LayerNorm 1 (fp32 math, fp16 rounding like PyTorch half layer_norm)
    mu = tl.sum(x, axis=0) / nf
    xc = tl.where(mask, x - mu, 0.0)
    var = tl.sum(xc * xc, axis=0) / nf
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g1 + b1
    y = y.to(tl.float16).to(tl.float32)  # round to fp16 between stages

    # LayerNorm 2
    mu2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / nf
    yc = tl.where(mask, y - mu2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / nf
    rstd2 = 1.0 / tl.sqrt(var2 + EPS_LN)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = yc * rstd2 * g2 + b2
    z = z.to(tl.float16).to(tl.float32)

    # Scale (fp32 opmath, rounded to fp16 like PyTorch)
    z = (z * SCALE).to(tl.float16).to(tl.float32)

    # RMSNorm: fp32 mean of squares, normalize in fp32, cast fp16, mul weight
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / nf
    inv = 1.0 / tl.sqrt(ms + EPS_RMS)
    zn = (z * inv).to(tl.float16).to(tl.float32)
    w = tl.load(RW + cols, mask=mask, other=0.0).to(tl.float32)
    out = (zn * w).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        y = y.contiguous()
        out = torch.empty_like(y)
        n_rows, n_cols = y.shape
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_norm_kernel[(n_rows,)](
            y, out,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.rms4_w,
            n_cols, y.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6, SCALE=1.1137,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
