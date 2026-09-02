import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 717
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    G0, B0, G1, B1, W2, G3, B3, W4,
    D: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 compute, cast to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    x = xc * tl.math.rsqrt(var + EPS_LN) * g0 + b0
    x = x.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = xc * tl.math.rsqrt(var + EPS_LN) * g1 + b1
    x = x.to(tl.float16).to(tl.float32)

    # ---- RMSNorm 2 (normalize in fp32, cast to fp16, then fp16 mul by w) ----
    ms = tl.sum(x * x, axis=0) / D
    xn = (x * tl.math.rsqrt(ms + EPS_RMS)).to(tl.float16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    x = (xn * w2).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = xc * tl.math.rsqrt(var + EPS_LN) * g3 + b3
    x = x.to(tl.float16).to(tl.float32)

    # ---- RMSNorm 4 ----
    ms = tl.sum(x * x, axis=0) / D
    xn = (x * tl.math.rsqrt(ms + EPS_RMS)).to(tl.float16)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0)
    out = xn * w4

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_norm_kernel[(n_rows,)](
            xc, y,
            self.ln0_g, self.ln0_b,
            self.ln1_g, self.ln1_b,
            self.rms2_w,
            self.ln3_g, self.ln3_b,
            self.rms4_w,
            D=d,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
        x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        return x
