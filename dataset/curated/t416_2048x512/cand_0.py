import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 416
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, Y,
    G0, B0, W1, G2, B2, W3,
    N_COLS: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    x = tl.load(X + row * N_COLS + offs, mask=mask, other=0.0).to(tl.float32)

    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)

    n = N_COLS.to(tl.float32) if False else N_COLS  # constexpr int

    # ---- LayerNorm 0 (fp32 compute, bf16 output rounding) ----
    mean = tl.sum(x, axis=0) / N_COLS
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N_COLS
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    y = xc * rstd * g0 + b0
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 1 ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N_COLS
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    y = (y * r).to(tl.bfloat16).to(tl.float32)
    y = (y * w1).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N_COLS
    yc = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N_COLS
    rstd2 = 1.0 / tl.sqrt(var2 + EPS_LN)
    y = yc * rstd2 * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 3 ----
    ms2 = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N_COLS
    r2 = 1.0 / tl.sqrt(ms2 + EPS_RMS)
    y = (y * r2).to(tl.bfloat16).to(tl.float32)
    y = (y * w3).to(tl.bfloat16)

    tl.store(Y + row * N_COLS + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)
        orig_shape = x.shape
        d = orig_shape[-1]
        xr = x.contiguous().view(-1, d)
        rows = xr.shape[0]
        out = torch.empty_like(xr)
        BLOCK = triton.next_power_of_2(d)
        _fused_norms_kernel[(rows,)](
            xr, out,
            self.ln0_g, self.ln0_b, self.rms1_w,
            self.ln2_g, self.ln2_b, self.rms3_w,
            N_COLS=d,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out.view(orig_shape)

    def _forward_ref(self, x):
        x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        return x
