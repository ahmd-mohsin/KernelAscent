import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 288
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, B1_ptr, W2_ptr, G4_ptr, B4_ptr, W5_ptr, OUT_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- load matmul output row, add bias (fp32 math, round to bf16) ----
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = (x + b1).to(tl.bfloat16)
    sf = s.to(tl.float32)

    # ---- RMSNorm #1 ----
    ms = tl.sum(sf * sf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    n1 = (sf * r).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = (n1 * w2).to(tl.bfloat16)
    x1f = x1.to(tl.float32)

    # ---- exact GELU (erf) in fp32, round to bf16 ----
    gel = x1f * 0.5 * (1.0 + tl.math.erf(x1f * 0.7071067811865476))
    g = gel.to(tl.bfloat16)
    gf = g.to(tl.float32)

    # ---- LayerNorm (fp32 internals, single rounding at end) ----
    mean = tl.sum(gf, axis=0) / N
    d = tl.where(mask, gf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    gam = tl.load(G4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bet = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (d * inv * gam + bet).to(tl.bfloat16)
    lnf = ln.to(tl.float32)

    # ---- RMSNorm #2 ----
    ms2 = tl.sum(lnf * lnf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    n2 = (lnf * r2).to(tl.bfloat16).to(tl.float32)
    w5 = tl.load(W5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (n2 * w5).to(tl.bfloat16)

    tl.store(OUT_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores)
        y = torch.matmul(x, self.W0)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            y, self.b1, self.rms2_w, self.ln4_g, self.ln4_b, self.rms5_w, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
