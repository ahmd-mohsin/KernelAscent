import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 216
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _ln_rms_kernel(
    X, LN_G, LN_B, RMS_W, OUT,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # round to bf16 (matches layer_norm output dtype), then upcast for RMS
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rr = 1.0 / tl.sqrt(ms + RMS_EPS)

    z = (yf * rr).to(tl.bfloat16).to(tl.float32)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_rms_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N, h.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
