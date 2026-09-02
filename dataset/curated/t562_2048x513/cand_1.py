import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 562
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_norm_gelu_kernel(
    X, OUT, RMS0, B1, G2, B2, RMS3,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    base = row * D

    x = tl.load(X + base + cols, mask=mask, other=0.0)

    # ---- RMSNorm 0 (fp32 accumulate, cast to fp16, then fp16 weight mul) ----
    xf = x.to(tl.float32)
    ms0 = tl.sum(xf * xf, axis=0) / D
    y = (xf * (1.0 / tl.sqrt(ms0 + 1e-6))).to(tl.float16)
    w0 = tl.load(RMS0 + cols, mask=mask, other=0.0)
    y = y * w0

    # ---- bias add (fp16) ----
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    y = y + b1

    # ---- LayerNorm (fp32 internal math, fp16 output) ----
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / D
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (diff * (1.0 / tl.sqrt(var + 1e-5)) * g2 + b2).to(tl.float16)

    # ---- RMSNorm 3 ----
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    ms3 = tl.sum(zf * zf, axis=0) / D
    u = (zf * (1.0 / tl.sqrt(ms3 + 1e-6))).to(tl.float16)
    w3 = tl.load(RMS3 + cols, mask=mask, other=0.0)
    u = u * w3

    # ---- exact GELU (erf, fp32 internal math) ----
    uf = u.to(tl.float32)
    out = uf * 0.5 * (1.0 + tl.math.erf(uf * 0.7071067811865476))

    tl.store(OUT + base + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = y + self.b1
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            _yf = y.float(); y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            return F.gelu(y)

        d = x.shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_norm_gelu_kernel[(rows,)](
            x2, out,
            self.rms0_w, self.b1, self.ln2_g, self.ln2_b, self.rms3_w,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(x.shape)
