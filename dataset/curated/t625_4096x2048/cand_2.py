import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 625
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, Y,
    G1, B1, W3, G4, B4,
    N,
    SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 internal math, fp16 output, eps=1e-5) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    h = (xc * rstd * g1 + b1).to(tl.float16)

    # ---- scalar multiply (opmath = fp32, result fp16) ----
    h = (h.to(tl.float32) * SCALE).to(tl.float16)

    # ---- RMSNorm (explicit fp32 math per reference, eps=1e-6) ----
    hf = h.to(tl.float32)
    ms = tl.sum(hf * hf, axis=0) / N
    rr = 1.0 / tl.sqrt(ms + 1e-6)
    h = (hf * rr).to(tl.float16)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    h = (h.to(tl.float32) * w3).to(tl.float16)

    # ---- LayerNorm 4 (fp32 internal math, fp16 output, eps=1e-5) ----
    hf = h.to(tl.float32)
    mean2 = tl.sum(hf, axis=0) / N
    hc = tl.where(mask, hf - mean2, 0.0)
    var2 = tl.sum(hc * hc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (hc * rstd2 * g4 + b4).to(tl.float16)

    tl.store(Y + row * N + offs, y, mask=mask)


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
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_kernel[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.rms3_w, self.ln4_g, self.ln4_b,
            N, 1.4972,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
