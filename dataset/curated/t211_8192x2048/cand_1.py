import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 211
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, Y,
    LN0_G, LN0_B, RMS2_W, LN3_G, LN3_B,
    stride_xm, stride_ym,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, round to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g0 = tl.load(LN0_G + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(LN0_B + cols, mask=mask, other=0.0).to(tl.float32)
    h = (xc * rstd) * g0 + b0
    h = h.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 math, round to fp16) ----
    h_for_max = tl.where(mask, h, float('-inf'))
    mx = tl.max(h_for_max, axis=0)
    e = tl.exp(h - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32, cast to fp16, mul by fp16 weight) ----
    ms = tl.sum(p * p, axis=0) / D
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    r = (p * rrms).to(tl.float16).to(tl.float32)
    w = tl.load(RMS2_W + cols, mask=mask, other=0.0).to(tl.float32)
    r = (r * w).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean3 = tl.sum(tl.where(mask, r, 0.0), axis=0) / D
    rc = tl.where(mask, r - mean3, 0.0)
    var3 = tl.sum(rc * rc, axis=0) / D
    rstd3 = 1.0 / tl.sqrt(var3 + 1e-5)
    g3 = tl.load(LN3_G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(LN3_B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (rc * rstd3) * g3 + b3

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


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
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x

        orig_shape = x.shape
        Dn = orig_shape[-1]
        x2 = x.contiguous().view(-1, Dn)
        Mn = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dn)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_norms_kernel[(Mn,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.rms2_w, self.ln3_g, self.ln3_b,
            x2.stride(0), y.stride(0),
            D=Dn, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
