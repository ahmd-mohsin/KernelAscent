import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 813
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, OUT_ptr,
    RMS1_ptr, B2_ptr, LN3G_ptr, LN3B_ptr, RMS4_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # ---- RMSNorm 1 (compute in fp32, round to bf16 like reference) ----
    ms = tl.sum(xf * xf, axis=0) / N
    y = xf * tl.math.rsqrt(ms + 1e-6)
    y = y.to(tl.bfloat16)  # cast point in reference

    w1 = tl.load(RMS1_ptr + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # ---- add bias b2 (bf16 op -> fp32 compute, bf16 round) ----
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    # ---- LayerNorm (fp32 stats, bf16 output) ----
    yf = y.to(tl.float32)
    mean = tl.sum(tl.where(mask, yf, 0.0), axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(LN3G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN3B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (diff * inv) * g + b
    z = z.to(tl.bfloat16)

    # ---- RMSNorm 4 ----
    zf = z.to(tl.float32)
    ms2 = tl.sum(zf * zf, axis=0) / N
    u = zf * tl.math.rsqrt(ms2 + 1e-6)
    u = u.to(tl.bfloat16)
    w4 = tl.load(RMS4_ptr + cols, mask=mask, other=0.0)
    u = (u.to(tl.float32) * w4.to(tl.float32)).to(tl.bfloat16)

    # ---- Softmax (fp32 compute, bf16 output) ----
    uf = u.to(tl.float32)
    uf = tl.where(mask, uf, float('-inf'))
    m = tl.max(uf, axis=0)
    e = tl.exp(uf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_post_kernel[(m,)](
            x, out,
            self.rms1_w, self.b2, self.ln3_g, self.ln3_b, self.rms4_w,
            n, x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
