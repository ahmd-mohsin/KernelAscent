import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 522
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X, W, G, B, Y,
    D: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- RMSNorm (fp32 math, round to fp16 like reference) ----
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + EPS_RMS)
    y16 = (x * r).to(tl.float16)                       # .to(x.dtype)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    z16 = (y16.to(tl.float32) * w).to(tl.float16)      # * rms0_w (fp16 op)
    s16 = (z16.to(tl.float32) * SCALE).to(tl.float16)  # * 1.3348 (fp32 opmath, fp16 out)

    # ---- LayerNorm (fp32 accumulation, fp16 output) ----
    sf = tl.where(mask, s16.to(tl.float32), 0.0)
    mean = tl.sum(sf, axis=0) / D
    diff = tl.where(mask, sf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = tl.math.rsqrt(var + EPS_LN)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((sf - mean) * inv * g + b).to(tl.float16)
    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x * 1.3348
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_rms_ln_kernel[(n_rows,)](
            xc, self.rms0_w, self.ln2_g, self.ln2_b, y,
            D=d, EPS_RMS=1e-6, EPS_LN=1e-5, SCALE=1.3348,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
