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
    X, OUT,
    W0, G1, B1, W2, B3,
    N, D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms = tl.sum(x * x, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)
    h = (x * r).to(tl.bfloat16)  # cast to bf16 as in reference
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    h = (h.to(tl.float32) * w0).to(tl.bfloat16)  # bf16 multiply (correctly rounded)
    xf = h.to(tl.float32)

    # ---- LayerNorm ----
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / D_
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xf - mean) * rstd * g1 + b1).to(tl.bfloat16)
    yf = y.to(tl.float32)

    # ---- RMSNorm 2 ----
    ms2 = tl.sum(yf * yf, axis=0) / D_
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    z = (yf * r2).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) * w2).to(tl.bfloat16)

    # ---- bias add ----
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z.to(tl.float32) + b3).to(tl.bfloat16)

    tl.store(OUT + row * D_ + cols, out, mask=mask)


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
        x2 = x.contiguous().view(-1, d)
        n = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_norm_kernel[(n,)](
            x2, out,
            self.rms0_w, self.ln1_g, self.ln1_b, self.rms2_w, self.b3,
            n, d,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
