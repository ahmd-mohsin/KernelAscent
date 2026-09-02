import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 892
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(x_ptr, w1_ptr, w2_ptr, out_ptr,
                  n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * n_cols

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # ---- load input (fp16 -> fp32 compute, matching PyTorch opmath) ----
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- gelu #1 (exact erf), round to fp16 like the fp16 tensor result ----
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g16 = g.to(tl.float16)

    # ---- RMSNorm #1 (fp32 accumulate, cast to fp16, then * weight) ----
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)
    ms = tl.sum(gf * gf, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (gf * inv).to(tl.float16)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y16 = (y16.to(tl.float32) * w1).to(tl.float16)

    # ---- gelu #2 ----
    yf = y16.to(tl.float32)
    g2 = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    g2_16 = g2.to(tl.float16)

    # ---- softmax (fp32 compute, fp16 output, like PyTorch half softmax) ----
    z = g2_16.to(tl.float32)
    z_masked = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z_masked, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    ssum = tl.sum(e, axis=0)
    sm16 = (e / ssum).to(tl.float16)

    # ---- RMSNorm #2 ----
    sf = sm16.to(tl.float32)
    sf = tl.where(mask, sf, 0.0)
    ms2 = tl.sum(sf * sf, axis=0) / n_cols
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    o16 = (sf * inv2).to(tl.float16)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (o16.to(tl.float32) * w2).to(tl.float16)

    tl.store(out_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference implementation
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(n_rows,)](
            x, self.rms1_w, self.rms4_w, out,
            n_cols, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
