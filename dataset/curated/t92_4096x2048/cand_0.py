import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 92
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_ln_ln_rms_relu(
    X, OUT, LN1G, LN1B, B2, LN3G, LN3B, RMSW,
    N, EPS_LN, EPS_RMS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load row (fp16 -> fp32 for reductions) ----
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 accumulation, fp16 output like PyTorch) ----
    mean1 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(xc * xc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS_LN)
    g1 = tl.load(LN1G + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(LN1B + offs, mask=mask, other=0.0).to(tl.float32)
    y16 = ((xc * rstd1) * g1 + b1).to(tl.float16)

    # ---- add bias (fp16 arithmetic, matching PyTorch) ----
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    y16 = y16 + b2

    # ---- LayerNorm 2 (fp32 accumulation, fp16 output) ----
    yf = y16.to(tl.float32)
    mean2 = tl.sum(yf, axis=0) / N
    yc = tl.where(mask, yf - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS_LN)
    g3 = tl.load(LN3G + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(LN3B + offs, mask=mask, other=0.0).to(tl.float32)
    z16 = ((yc * rstd2) * g3 + b3).to(tl.float16)

    # ---- RMSNorm in fp32, cast to fp16, multiply by weight in fp16 ----
    zf = z16.to(tl.float32)
    ms = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    w = tl.load(RMSW + offs, mask=mask, other=0.0)
    out = (zf * r).to(tl.float16) * w

    # ---- ReLU ----
    zero = tl.full(out.shape, 0.0, tl.float16)
    out = tl.maximum(out, zero)

    tl.store(OUT + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_rms_relu[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.b2,
            self.ln3_g, self.ln3_b, self.rms4_w,
            N, 1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
