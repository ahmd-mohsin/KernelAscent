import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 92
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_post_kernel(
    X, Y,
    LN1G, LN1B, B2, LN3G, LN3B, RMSW,
    N, stride_x, stride_y,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 internal, fp16 output) ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS_LN)
    g1 = tl.load(LN1G + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(LN1B + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = (d1 * rstd1 * g1 + b1).to(tl.float16)

    # ---- add b2 (fp16 arithmetic) ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    y2 = y1 + b2

    # ---- LayerNorm 3 (fp32 internal, fp16 output) ----
    xf = y2.to(tl.float32)
    mean3 = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d3 = tl.where(mask, xf - mean3, 0.0)
    var3 = tl.sum(d3 * d3, axis=0) / N
    rstd3 = 1.0 / tl.sqrt(var3 + EPS_LN)
    g3 = tl.load(LN3G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(LN3B + cols, mask=mask, other=0.0).to(tl.float32)
    y3 = (d3 * rstd3 * g3 + b3).to(tl.float16)

    # ---- RMSNorm (fp32 compute, cast to fp16, multiply by weight in fp16) ----
    xr = y3.to(tl.float32)
    ms = tl.sum(tl.where(mask, xr * xr, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    y4 = (xr * rrms).to(tl.float16)
    w = tl.load(RMSW + cols, mask=mask, other=0.0)
    y5 = y4 * w

    # ---- ReLU ----
    zero = tl.zeros(y5.shape, dtype=tl.float16)
    out = tl.maximum(y5, zero)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


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
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_post_kernel[(m,)](
            h, out,
            self.ln1_g, self.ln1_b, self.b2, self.ln3_g, self.ln3_b, self.rms4_w,
            n, h.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
