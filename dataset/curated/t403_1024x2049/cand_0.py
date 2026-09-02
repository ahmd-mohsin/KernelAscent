import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 403
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _fused_kernel(
    X, W_RMS, G_LN, B_LN, OUT,
    D: tl.constexpr,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, erf), computed in fp32 then rounded to fp16 (matches PyTorch CUDA half gelu)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g16 = g.to(tl.float16)

    # ---- RMSNorm: _xf = x.float(); x = (_xf * rsqrt(mean(_xf^2)+1e-6)).half() * w
    xf = g16.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / D
    rrms = tl.math.rsqrt(ms + 1e-6)
    y16 = (xf * rrms).to(tl.float16)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    z32 = y16.to(tl.float32) * w.to(tl.float32)
    z16 = z32.to(tl.float16)

    # ---- ReLU (in fp16)
    z16 = tl.maximum(z16, tl.zeros_like(z16))

    # ---- LayerNorm (fp32 internal math, fp16 output)
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean = tl.sum(zf, axis=0) / D
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = tl.math.rsqrt(var + 1e-5)

    gamma = tl.load(G_LN + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_LN + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd) * gamma + beta

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms1_w, self.ln3_g, self.ln3_b, out,
            d, x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
