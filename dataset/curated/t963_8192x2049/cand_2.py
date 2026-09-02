import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 963
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _fused_kernel(
    X, W_RMS, G_LN, B_LN, OUT,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # step 1: scale, round to fp16
    y = (x * SCALE).to(tl.float16)

    # step 2: RMSNorm in fp32
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = tl.math.rsqrt(ms + EPS_RMS)
    t = (yf * r).to(tl.float16)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0).to(tl.float32)
    a = (t.to(tl.float32) * w).to(tl.float16)

    # step 3: LayerNorm (fp32 accumulation)
    af = a.to(tl.float32)
    af = tl.where(mask, af, 0.0)
    mean = tl.sum(af, axis=0) / N
    diff = tl.where(mask, af - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.math.rsqrt(var + EPS_LN)

    g = tl.load(G_LN + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_LN + cols, mask=mask, other=0.0).to(tl.float32)
    ln = ((af - mean) * inv * g + b).to(tl.float16)

    # step 4: exact GELU in fp32
    lf = ln.to(tl.float32)
    ge = 0.5 * lf * (1.0 + tl.math.erf(lf * 0.7071067811865476))
    out = ge.to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, out,
            N, x.stride(0), out.stride(0),
            SCALE=1.3694,
            EPS_RMS=1e-6,
            EPS_LN=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
