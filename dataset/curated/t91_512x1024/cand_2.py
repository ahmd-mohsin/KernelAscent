import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 91
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_kernel(
    X, Y, G, B, W3, W4,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, round once to fp16) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h = (xc * rstd) * g + b
    h = h.to(tl.float16)

    # ---- GELU (exact, erf; fp32 opmath, round to fp16) ----
    hf = h.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = 0.5 * hf * (1.0 + tl.math.erf(hf * INV_SQRT2))
    gel = gel.to(tl.float16)

    # ---- scale by 1.0939 (fp32 opmath, round to fp16) ----
    s = (gel.to(tl.float32) * 1.0939).to(tl.float16)

    # ---- RMSNorm 1 ----
    sf = s.to(tl.float32)
    sf = tl.where(mask, sf, 0.0)
    ms = tl.sum(sf * sf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    t = (sf * r).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    t = (t.to(tl.float32) * w3.to(tl.float32)).to(tl.float16)

    # ---- RMSNorm 2 ----
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, 0.0)
    ms2 = tl.sum(tf * tf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + EPS_RMS)
    u = (tf * r2).to(tl.float16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0)
    u = (u.to(tl.float32) * w4.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * stride_y + cols, u, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.gelu(x)
            x = x * 1.0939
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_kernel[(Mrows,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.rms3_w, self.rms4_w,
            x2.stride(0), y.stride(0),
            N=N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
