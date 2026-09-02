import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 826
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, OUT,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch half layer_norm)
    mean = tl.sum(x, axis=0) / D_
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D_
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    # round to fp16 (layer_norm output dtype)
    y16 = y.to(tl.float16)

    # x * 1.1102 : computed with fp32 opmath, result fp16
    y16 = (y16.to(tl.float32) * 1.1102).to(tl.float16)

    # RMSNorm in fp32
    xf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)
    z16 = (xf * r).to(tl.float16)

    # multiply by rms weight (fp32 opmath, fp16 result)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z16.to(tl.float32) * w).to(tl.float16)

    tl.store(OUT + row * D_ + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_ln_rms_kernel[(Mrows,)](
            x, self.ln0_g, self.ln0_b, self.rms2_w, out,
            D_=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
