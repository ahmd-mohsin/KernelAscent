import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 414
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, OUT,
    RMS_W, G2, B2, G3, B3,
    N, stride_x, stride_o,
    EPS_RMS: tl.constexpr, EPS_LN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (fp32 compute, round to fp16, then * weight in fp32, round to fp16) ----
    ms = tl.sum(x * x, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + EPS_RMS)
    y = (x * rs).to(tl.float16)  # round to fp16 like reference
    w = tl.load(RMS_W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w).to(tl.float16)

    # ---- LayerNorm 2 ----
    yf = y.to(tl.float32)
    mean2 = tl.sum(yf, axis=0) / N
    d2 = tl.where(mask, yf - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS_LN)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = (d2 * rstd2 * g2 + b2).to(tl.float16)

    # ---- LayerNorm 3 ----
    y2f = y2.to(tl.float32)
    mean3 = tl.sum(y2f, axis=0) / N
    d3 = tl.where(mask, y2f - mean3, 0.0)
    var3 = tl.sum(d3 * d3, axis=0) / N
    rstd3 = 1.0 / tl.sqrt(var3 + EPS_LN)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y3 = (d3 * rstd3 * g3 + b3).to(tl.float16)

    # ---- final scale (fp32 opmath, output fp16) ----
    out = (y3.to(tl.float32) * SCALE).to(tl.float16)
    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_norms_kernel[(Mrows,)](
            x, out,
            self.rms1_w, self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            N, x.stride(0), out.stride(0),
            1e-6, 1e-5, 1.1262,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
