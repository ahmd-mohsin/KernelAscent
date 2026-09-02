import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 916
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_ln_relu_rms_gelu(
    Y, OUT, G, B, W,
    N, stride,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch half accumulate behavior)
    mean = tl.sum(y, axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h = d * rstd * g + b

    # cast to fp16 (layer_norm output dtype), relu, upcast to fp32
    h16 = h.to(tl.float16)
    r16 = tl.maximum(h16, tl.zeros_like(h16))
    rf = r16.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, rf * rf, 0.0), axis=0) / N
    rr = 1.0 / tl.sqrt(ms + EPS_RMS)
    n16 = (rf * rr).to(tl.float16)

    # multiply by rms weight (fp32 opmath, round to fp16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    m16 = (n16.to(tl.float32) * w).to(tl.float16)

    # exact GELU in fp32, output fp16
    gf = m16.to(tl.float32)
    out = 0.5 * gf * (1.0 + tl.math.erf(gf * 0.7071067811865476))

    tl.store(OUT + row * stride + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_relu_rms_gelu[(Mrows,)](
            y, out, self.ln1_g, self.ln1_b, self.rms3_w,
            N, y.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
