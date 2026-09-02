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
    X, Y, G1, B1, W3, G4, B4,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, fp16 rounding of output like PyTorch) ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    inv1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d1 * inv1 * g1 + b1).to(tl.float16).to(tl.float32)

    # ---- scale by 1.4972 (fp32 compute, round to fp16) ----
    y = (y * SCALE).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32) ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((y * rinv).to(tl.float16).to(tl.float32) * w3).to(tl.float16).to(tl.float32)
    z = tl.where(mask, z, 0.0)

    # ---- LayerNorm 4 ----
    mean2 = tl.sum(z, axis=0) / N
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * inv2 * g4 + b4

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


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
            N, h.stride(0), out.stride(0),
            SCALE=1.4972,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
