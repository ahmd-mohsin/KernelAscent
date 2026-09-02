import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 621
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_ln_relu_ln_rms_gelu(
    X, G0, B0, G2, B2, W3, OUT,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    base = row * D

    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 compute, round once to bf16) ----
    mean0 = tl.sum(x, axis=0) / D
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / D
    inv0 = tl.math.rsqrt(var0 + 1e-5)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d0 * inv0 * g0 + b0
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU ----
    y = tl.maximum(y, 0.0)

    # ---- LayerNorm 2 (fp32 compute, round once to bf16) ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / D
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / D
    inv2 = tl.math.rsqrt(var2 + 1e-5)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * inv2 * g2 + b2
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32 compute, round to bf16, then bf16*bf16 mul via fp32 opmath) ----
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    zn = (z * r).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    u = (zn * w3).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact erf, fp32 opmath) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * u * (1.0 + tl.math.erf(u * INV_SQRT2))

    tl.store(OUT + base + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = torch.relu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            yf = y.float()
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            return F.gelu(y)

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        rows = xc.shape[0]
        out = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _fused_ln_relu_ln_rms_gelu[(rows,)](
            xc, self.ln0_g, self.ln0_b, self.ln2_g, self.ln2_b, self.rms3_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
