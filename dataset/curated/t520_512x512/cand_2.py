import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 520
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, W0, W3, G, B, Y,
                  n_rows,
                  D: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- load input, upcast to fp32 (matches x.float()) ----
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xb = (x * r).to(tl.bfloat16).to(tl.float32)          # .to(bf16)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xb * w0).to(tl.bfloat16).to(tl.float32)         # bf16 * bf16 -> bf16 (fp32 opmath)

    # ---- scalar scale ----
    x = (x * 1.1232).to(tl.bfloat16).to(tl.float32)

    # ---- softmax (fp32 accumulation, bf16 output) ----
    xm = tl.where(mask, x, float("-inf"))
    mmax = tl.max(xm, axis=0)
    e = tl.exp(xm - mmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 3 ----
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xb = (x * r).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xb * w3).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 math, bf16 output) ----
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd * g + b).to(tl.bfloat16)

    tl.store(Y + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(n_rows,)](
            x2, self.rms0_w, self.rms3_w, self.ln4_g, self.ln4_b, y,
            n_rows, D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
