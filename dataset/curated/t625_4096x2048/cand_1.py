import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 625
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, G1, B1, W3, G4, B4, Out,
    N,
    stride_x, stride_o,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    n = N.to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, output rounded to fp16 like reference) ----
    mean1 = tl.sum(x, axis=0) / n
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n
    inv1 = 1.0 / tl.sqrt(var1 + EPS_LN)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d1 * inv1 * g1 + b1
    y = y.to(tl.float16).to(tl.float32)

    # ---- scale by 1.4972 (fp32 compute, rounded to fp16) ----
    y = (y * SCALE).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32) ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / n
    rinv = 1.0 / tl.sqrt(ms + EPS_RMS)
    z = (y * rinv).to(tl.float16).to(tl.float32)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z * w3).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 4 ----
    mean4 = tl.sum(tl.where(mask, z, 0.0), axis=0) / n
    d4 = tl.where(mask, z - mean4, 0.0)
    var4 = tl.sum(d4 * d4, axis=0) / n
    inv4 = 1.0 / tl.sqrt(var4 + EPS_LN)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = d4 * inv4 * g4 + b4

    tl.store(Out + row * stride_o + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (same as reference)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_norm_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms3_w, self.ln4_g, self.ln4_b, out,
            N,
            h.stride(0), out.stride(0),
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            SCALE=1.4972,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
