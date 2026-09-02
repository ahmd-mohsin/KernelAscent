import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 526
M, D, DT = 1024, 1025, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, W1, W2, G3, B3, G4, B4, Y,
    D: tl.constexpr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, bf16 output) ----
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    xb = (e / s).to(tl.bfloat16)

    # ---- RMSNorm 1 ----
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, 0) / D
    r = tl.rsqrt(ms + 1e-6)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    xb = (xf * r).to(tl.bfloat16) * w1

    # ---- RMSNorm 2 ----
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, 0) / D
    r = tl.rsqrt(ms + 1e-6)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    xb = (xf * r).to(tl.bfloat16) * w2

    # ---- LayerNorm 3 ----
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, 0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, 0) / D
    rstd = tl.rsqrt(var + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    xb = (diff * rstd * g3 + b3).to(tl.bfloat16)

    # ---- LayerNorm 4 ----
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, 0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, 0) / D
    rstd = tl.rsqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    xb = (diff * rstd * g4 + b4).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, xb, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            x = torch.softmax(x, dim=-1)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_norm_kernel[(m,)](
            x2, self.rms1_w, self.rms2_w,
            self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b,
            y,
            d,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
