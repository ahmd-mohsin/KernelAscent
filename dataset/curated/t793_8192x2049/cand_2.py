import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 793
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _fused_post_kernel(
    X, W_RMS, G4, B4, G5, B5, OUT,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, fp16 output like PyTorch) ----
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # ---- exact GELU (erf form, fp32 math, fp16 output) ----
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32) then fp16 multiply with weight ----
    ms = tl.sum(g * g, 0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y16 = (g * r).to(tl.float16)
    w = tl.load(W_RMS + offs, mask=mask, other=0.0)  # fp16
    y = (y16 * w).to(tl.float32)  # fp16 multiply, then upcast

    # ---- LayerNorm 4 (fp32 internal, fp16 output) ----
    mean = tl.sum(tl.where(mask, y, 0.0), 0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g4 + b4).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 5 (fp32 internal, fp16 output) ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), 0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / N
    rstd2 = tl.math.rsqrt(var2 + 1e-5)
    g5 = tl.load(G5 + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (d2 * rstd2 * g5 + b5).to(tl.float16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty((Mrows, N), dtype=h.dtype, device=h.device)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            h, self.rms3_w, self.ln4_g, self.ln4_b, self.ln5_g, self.ln5_b, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
