import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 802
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, Y, W0, G1, B1, W2, B3,
    stride_x, stride_y,
    D_: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # ---- RMSNorm 0 ----
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = tl.math.rsqrt(ms + 1e-6)
    x1 = (xf * inv).to(tl.bfloat16)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    x1 = (x1.to(tl.float32) * w0.to(tl.float32)).to(tl.bfloat16)

    # ---- LayerNorm ----
    x1f = x1.to(tl.float32)
    mean = tl.sum(x1f, axis=0) / D_
    diff = tl.where(mask, x1f - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    rstd = tl.math.rsqrt(var + 1e-5)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = (diff * rstd * g1 + b1).to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    x2f = x2.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, x2f * x2f, 0.0), axis=0) / D_
    inv2 = tl.math.rsqrt(ms2 + 1e-6)
    x3 = (x2f * inv2).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    x3 = (x3.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # ---- bias add ----
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    out = (x3.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x + self.b3

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_norm_kernel[(m,)](
            x2d, y,
            self.rms0_w, self.ln1_g, self.ln1_b, self.rms2_w, self.b3,
            x2d.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
