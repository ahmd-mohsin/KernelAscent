import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 751
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X, B1, W2, W3, G4, B4, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, bf16 output like PyTorch) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # ---- add bias (bf16) ----
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.bfloat16)
    x1 = (sm + b1).to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    xf = x1.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.bfloat16)
    x2 = ((xf * r).to(tl.bfloat16) * w2).to(tl.bfloat16)

    # ---- RMSNorm 3 ----
    xf = x2.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.bfloat16)
    x3 = ((xf * r).to(tl.bfloat16) * w3).to(tl.bfloat16)

    # ---- LayerNorm (fp32 internal, bf16 output like PyTorch) ----
    xf = x3.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / D
    xc = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xc * inv * g + b).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            xx = torch.softmax(x, dim=-1)
            xx = xx + self.b1
            _xf = xx.float()
            xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms2_w
            _xf = xx.float()
            xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms3_w
            return F.layer_norm(xx, (xx.shape[-1],), self.ln4_g, self.ln4_b)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_row_kernel[(n_rows,)](
            x2d, self.b1, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b, y,
            x2d.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
