import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 211
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    LN0G, LN0B, RMSW, LN3G, LN3B,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, fp16 rounding of output) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g0 = tl.load(LN0G + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(LN0B + cols, mask=mask, other=0.0).to(tl.float32)
    h = (xc * rstd) * g0 + b0
    h = h.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 math, fp16 output) ----
    hm = tl.where(mask, h, float('-inf'))
    row_max = tl.max(hm, axis=0)
    e = tl.exp(hm - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom
    s16 = s.to(tl.float16)

    # ---- RMSNorm (fp32 math on fp16 input, fp16 mul with weight) ----
    sf = s16.to(tl.float32)
    ms = tl.sum(tl.where(mask, sf * sf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(RMSW + cols, mask=mask, other=0.0)
    r16 = (sf * rrms).to(tl.float16) * w  # fp16 multiply, matches PyTorch
    r = r16.to(tl.float32)

    # ---- LayerNorm 3 ----
    mean2 = tl.sum(tl.where(mask, r, 0.0), axis=0) / N
    rc = tl.where(mask, r - mean2, 0.0)
    var2 = tl.sum(rc * rc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g3 = tl.load(LN3G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(LN3B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (rc * rstd2) * g3 + b3

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2d = x.contiguous().view(-1, n)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n)
        _fused_norm_kernel[(m,)](
            x2d, y,
            self.ln0_g, self.ln0_b, self.rms2_w, self.ln3_g, self.ln3_b,
            x2d.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
