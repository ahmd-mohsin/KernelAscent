import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 150
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, OUT,
    G, B, W3, W4, W5,
    N, stride_x, stride_o,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (float compute, bf16 output) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- layer norm (float compute, bf16 output) ----
    n_f = N.to(tl.float32)
    mean = tl.sum(y, axis=0) / n_f
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    inv = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((y - mean) * inv * g + b).to(tl.bfloat16)

    # ---- RMS norm 3 ----
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    r = 1.0 / tl.sqrt(tl.sum(zf * zf, axis=0) / n_f + EPS_RMS)
    t = (zf * r).to(tl.bfloat16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    z = t * w3

    # ---- RMS norm 4 ----
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    r = 1.0 / tl.sqrt(tl.sum(zf * zf, axis=0) / n_f + EPS_RMS)
    t = (zf * r).to(tl.bfloat16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    z = t * w4

    # ---- RMS norm 5 ----
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    r = 1.0 / tl.sqrt(tl.sum(zf * zf, axis=0) / n_f + EPS_RMS)
    t = (zf * r).to(tl.bfloat16)
    w5 = tl.load(W5 + cols, mask=mask, other=0.0).to(tl.bfloat16)
    z = t * w5

    tl.store(OUT + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(Mrows,)](
            h, out,
            self.ln2_g, self.ln2_b, self.rms3_w, self.rms4_w, self.rms5_w,
            N, h.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
