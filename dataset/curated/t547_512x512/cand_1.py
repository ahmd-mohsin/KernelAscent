import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 547
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X, B2, G3, Bt3, B4, W5, Out,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU, rounded to bf16 like PyTorch elementwise op
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # + b2, rounded to bf16
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g + b2).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32
    n = N.to(tl.float32)
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / n
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n
    inv = tl.math.rsqrt(var + LN_EPS)
    gamma = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Bt3 + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (d * inv * gamma + beta).to(tl.bfloat16).to(tl.float32)

    # + b4, rounded to bf16
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (ln + b4).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 (matches reference: fp32 math, cast to bf16, then * w)
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / n
    rinv = tl.math.rsqrt(ms + RMS_EPS)
    zn = (z * rinv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W5 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (zn * w).to(tl.bfloat16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core bf16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            h, self.b2, self.ln3_g, self.ln3_b, self.b4, self.rms5_w, out,
            N, h.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
