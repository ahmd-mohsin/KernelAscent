import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 644
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_row_kernel(
    X, B1, G3, B3, G4, B4, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- softmax (fp32 math, fp16 output, matching PyTorch) ----
    x = tl.load(X + base + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    sm = (e / s).to(tl.float16)

    # ---- add bias (fp16 arithmetic, matching PyTorch half add) ----
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    h = sm + b1  # fp16

    # ---- gelu (fp32 opmath, fp16 output) ----
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # ---- layer norm 1 (fp32 stats/affine, fp16 output) ----
    xf = g16.to(tl.float32)
    mean1 = tl.sum(tl.where(mask, xf, 0.0), 0) / D
    d1 = tl.where(mask, xf - mean1, 0.0)
    var1 = tl.sum(d1 * d1, 0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    y1 = ((xf - mean1) * rstd1 * g3 + b3).to(tl.float16)

    # ---- layer norm 2 ----
    yf = y1.to(tl.float32)
    mean2 = tl.sum(tl.where(mask, yf, 0.0), 0) / D
    d2 = tl.where(mask, yf - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((yf - mean2) * rstd2 * g4 + b4).to(tl.float16)

    tl.store(Y + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if (not x.is_cuda) or x.dtype != torch.float16:
            # fallback: reference path
            x = torch.softmax(x, dim=-1)
            x = x + self.b1
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_row_kernel[(n_rows,)](
            x2, self.b1, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
