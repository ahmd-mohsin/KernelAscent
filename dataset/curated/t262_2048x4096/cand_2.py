import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 262
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, G0, B0, G2, B2, W3, Y,
    D_: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 accum, bf16 output rounding) ----
    mean0 = tl.sum(x, axis=0) / D_
    xm = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(xm * xm, axis=0) / D_
    rstd0 = 1.0 / tl.sqrt(var0 + 1e-5)
    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    y0 = (xm * rstd0 * g0 + b0).to(tl.bfloat16)

    # ---- scale by 1.3978 (fp32 opmath, bf16 rounding, like PyTorch) ----
    y1 = (y0.to(tl.float32) * 1.3978).to(tl.bfloat16)

    # ---- LayerNorm 2 (fp32 accum, bf16 output rounding) ----
    z = y1.to(tl.float32)
    z = tl.where(mask, z, 0.0)
    mean2 = tl.sum(z, axis=0) / D_
    zm = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(zm * zm, axis=0) / D_
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y2 = (zm * rstd2 * g2 + b2).to(tl.bfloat16)

    # ---- RMSNorm (fp32 math, bf16 rounding, then bf16*bf16 weight mul) ----
    xf = y2.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / D_
    rr = 1.0 / tl.sqrt(ms + 1e-6)
    y3 = (xf * rr).to(tl.bfloat16)

    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y3.to(tl.float32) * w3).to(tl.bfloat16)

    tl.store(Y + row * D_ + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_norms_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, self.ln2_g, self.ln2_b, self.rms3_w, y,
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
