import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 485
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_rms_gelu_relu_ln(
    X, W_RMS, G_LN, B_LN, OUT,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, rounded to bf16, then scaled by bf16 weight in fp32 math)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y_bf = (xf * r).to(tl.bfloat16)
    w = tl.load(W_RMS + offs, mask=mask, other=0.0)
    y_bf = (y_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # GELU (erf-based, fp32 opmath) -> bf16
    yf = y_bf.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)

    # ReLU (exact in bf16)
    g_bf = tl.maximum(g_bf, tl.zeros_like(g_bf))

    # LayerNorm (fp32 accumulation)
    v = g_bf.to(tl.float32)
    mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / N
    diff = tl.where(mask, v - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G_LN + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_LN + offs, mask=mask, other=0.0).to(tl.float32)
    out = (v - mean) * rstd * gamma + beta

    tl.store(OUT + row * N + offs, out.to(tl.bfloat16), mask=mask)


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
        _fused_rms_gelu_relu_ln[(m,)](
            x, self.rms1_w, self.ln4_g, self.ln4_b, out,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
