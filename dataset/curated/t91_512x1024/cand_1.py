import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 91
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_kernel(x_ptr, out_ptr,
                  g_ptr, b_ptr, w3_ptr, w4_ptr,
                  D: tl.constexpr, BLOCK: tl.constexpr,
                  SCALE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, biased variance, eps=1e-5)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd * g + b).to(tl.float16)  # round to fp16 like PyTorch output

    # GELU (erf-based) in fp32, round to fp16
    yf = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = (0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))).to(tl.float16)

    # scale
    s = (gel.to(tl.float32) * SCALE).to(tl.float16)

    # RMSNorm 1 (eps=1e-6)
    sf = s.to(tl.float32)
    ms1 = tl.sum(tl.where(mask, sf * sf, 0.0), axis=0) / D
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    n1 = (sf * r1).to(tl.float16)
    w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0)
    z1 = (n1.to(tl.float32) * w3.to(tl.float32)).to(tl.float16)

    # RMSNorm 2 (eps=1e-6)
    zf = z1.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    n2 = (zf * r2).to(tl.float16)
    w4 = tl.load(w4_ptr + cols, mask=mask, other=0.0)
    z2 = (n2.to(tl.float32) * w4.to(tl.float32)).to(tl.float16)

    tl.store(out_ptr + row * D + cols, z2, mask=mask)


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
            _xf = y.float(); y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            _xf = y.float(); y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms4_w
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, out,
            self.ln0_g, self.ln0_b, self.rms3_w, self.rms4_w,
            d, BLOCK, 1.0939,
            num_warps=8,
        )
        return out.view(orig_shape)
