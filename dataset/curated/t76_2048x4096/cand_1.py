import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 76
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_rms_gelu_ln_kernel(
    X, OUT, RW, G, B,
    N, stride_x, stride_o,
    EPS_RMS: tl.constexpr, EPS_LN: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, weight mul in fp16)
    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    xh = (x * rrms).to(tl.float16)
    rw = tl.load(RW + cols, mask=mask, other=0.0)
    xh = xh * rw

    # GELU (exact, erf) computed in fp32 (opmath), output fp16
    xf = xh.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    gh = g.to(tl.float16)

    # LayerNorm computed in fp32
    yf = gh.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS_LN)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    o = (yf - mean) * inv * gamma + beta
    oh = o.to(tl.float16)

    # scale (opmath fp32, output fp16)
    oh = (oh.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, oh, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_ln_kernel[(Mrows,)](
            x, out, self.rms1_w, self.ln3_g, self.ln3_b,
            N, x.stride(0), out.stride(0),
            1e-6, 1e-5, 1.1664,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
