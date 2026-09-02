import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 91
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_kernel(X, G, B, W3, W4, OUT,
                  N, eps_ln, eps_rms,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, cast to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    y = y.to(tl.float16)

    # ---- GELU (exact, fp32 opmath, cast back to fp16) ----
    yf = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865475
    gelu = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    gelu = gelu.to(tl.float16)

    # ---- scale by 1.0939 (fp32 opmath, cast back to fp16) ----
    s = (gelu.to(tl.float32) * 1.0939).to(tl.float16)

    # ---- RMSNorm #1 ----
    sf = s.to(tl.float32)
    sf = tl.where(mask, sf, 0.0)
    ms = tl.sum(sf * sf, axis=0) / N
    rrms = tl.math.rsqrt(ms + eps_rms)
    n1 = (sf * rrms).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    r1 = (n1.to(tl.float32) * w3).to(tl.float16)

    # ---- RMSNorm #2 ----
    rf = r1.to(tl.float32)
    rf = tl.where(mask, rf, 0.0)
    ms2 = tl.sum(rf * rf, axis=0) / N
    rrms2 = tl.math.rsqrt(ms2 + eps_rms)
    n2 = (rf * rrms2).to(tl.float16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.float32)
    r2 = (n2.to(tl.float32) * w4).to(tl.float16)

    tl.store(OUT + row * N + cols, r2, mask=mask)


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
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.gelu(y) * 1.0939
            yf = y.float()
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            yf = y.float()
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms4_w
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        xr = x.contiguous().view(-1, N)
        rows = xr.shape[0]
        out = torch.empty_like(xr)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(rows,)](
            xr, self.ln0_g, self.ln0_b, self.rms3_w, self.rms4_w, out,
            N, 1e-5, 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
