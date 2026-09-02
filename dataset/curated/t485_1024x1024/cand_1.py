import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 485
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_norm_act_kernel(
    X, RMS_W, LN_G, LN_B, OUT,
    N, stride_x, stride_o,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, cast to bf16, then bf16 multiply by weight)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)
    y = (xf * rrms).to(tl.bfloat16)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = (y * w)  # bf16 multiply, matches PyTorch semantics

    # GELU (erf variant) computed in fp32, cast back to bf16
    t = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = t * 0.5 * (1.0 + tl.math.erf(t * INV_SQRT2))
    g = g.to(tl.bfloat16)

    # ReLU
    g = tl.maximum(g, tl.zeros_like(g))

    # LayerNorm in fp32 accumulation
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)
    mean = tl.sum(gf, axis=0) / N
    diff = tl.where(mask, gf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    gamma = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (gf - mean) * rstd * gamma + beta

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_norm_act_kernel[(m,)](
            x, self.rms1_w, self.ln4_g, self.ln4_b, out,
            n, x.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
