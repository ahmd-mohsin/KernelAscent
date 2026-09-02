import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 211
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_kernel(X, Y, G0, B0, W2, G3, B3, N, stride,
                  EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    h = (d * rstd * g0 + b0).to(tl.float16)

    # ---- Softmax ----
    hf = tl.where(mask, h.to(tl.float32), float('-inf'))
    mx = tl.max(hf, axis=0)
    e = tl.exp(hf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # ---- RMSNorm ----
    xf = sm.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    r = ((xf * rrms).to(tl.float16) * w2).to(tl.float16)

    # ---- LayerNorm 3 ----
    rf = tl.where(mask, r.to(tl.float32), 0.0)
    mean3 = tl.sum(rf, axis=0) / N
    d3 = tl.where(mask, rf - mean3, 0.0)
    var3 = tl.sum(d3 * d3, axis=0) / N
    rstd3 = 1.0 / tl.sqrt(var3 + EPS_LN)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (d3 * rstd3 * g3 + b3).to(tl.float16)

    tl.store(Y + row * stride + cols, out, mask=mask)


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
        if not (x.is_cuda and x.dtype == torch.float16):
            h = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            h = torch.softmax(h, dim=-1)
            hf = h.float()
            h = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms2_w
            return F.layer_norm(h, (h.shape[-1],), self.ln3_g, self.ln3_b)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x2, y, self.ln0_g, self.ln0_b, self.rms2_w, self.ln3_g, self.ln3_b,
            n, x2.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
