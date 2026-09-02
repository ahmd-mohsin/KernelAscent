import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 511
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_row_kernel(
    X, W, G, B, Y,
    D: tl.constexpr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- load row (fp16 -> fp32) ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, matches torch half softmax) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)                      # masked lanes: exp(-inf) = 0
    s = tl.sum(e, axis=0)
    sm = e / s

    # cast to fp16 (softmax output dtype), then scale computed in fp32, cast back (torch opmath)
    x1 = sm.to(tl.float16)
    x2 = (x1.to(tl.float32) * 1.0652).to(tl.float16)

    # ---- RMSNorm (fp32) ----
    xf = x2.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    x3 = (xf * r).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)      # fp16
    x4 = x3 * w                                       # native fp16 mul (matches half*half)
    x5 = (x4.to(tl.float32) * 1.061).to(tl.float16)   # scalar mul in fp32 opmath

    # ---- LayerNorm (fp32 internal, like torch) ----
    xf2 = tl.where(mask, x5.to(tl.float32), 0.0)
    mean = tl.sum(xf2, axis=0) / D
    d = tl.where(mask, xf2 - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    inv = tl.math.rsqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((xf2 - mean) * inv * g + b).to(tl.float16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xs = torch.softmax(x, dim=-1) * 1.0652
            _xf = xs.float()
            xs = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xs.dtype) * self.rms2_w
            xs = xs * 1.061
            return F.layer_norm(xs, (xs.shape[-1],), self.ln4_g, self.ln4_b)

        x = x.contiguous()
        rows, d = x.shape[0], x.shape[-1]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_row_kernel[(rows,)](
            x, self.rms2_w, self.ln4_g, self.ln4_b, y,
            d, x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
