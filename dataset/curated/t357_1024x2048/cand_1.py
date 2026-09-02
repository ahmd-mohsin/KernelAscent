import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 357
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_softmax_rms_gelu(
    X, W, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # --- softmax (fp32 compute, output rounded to fp16 like torch) ---
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom
    s16 = s.to(tl.float16)          # torch.softmax returns fp16
    sf = s16.to(tl.float32)         # _xf = x.float()

    # --- RMSNorm ---
    ms = tl.sum(sf * sf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    y16 = (sf * r).to(tl.float16)   # .to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0)
    prod = (y16 * w).to(tl.float16)  # fp16 * fp16 -> fp16 (like torch)

    # --- GELU (torch computes in fp32 for half inputs) ---
    pf = prod.to(tl.float32)
    g = pf * 0.5 * (1.0 + tl.math.erf(pf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_softmax_rms_gelu[(Mrows,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            D=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
