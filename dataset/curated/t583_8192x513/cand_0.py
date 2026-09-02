import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 583
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _fused_post_kernel(X, LN_G, LN_B, RMS_W, OUT,
                       N, stride_x, stride_o,
                       eps_ln, eps_rms,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), computed in fp32, rounded to fp16 (matches F.gelu on half)
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    g = g.to(tl.float16).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch internal fp32 accumulation)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + eps_ln)

    ln_g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    ln_b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g - mean) * rstd * ln_g + ln_b
    y = y.to(tl.float16)

    # ReLU
    y = tl.maximum(y, tl.zeros_like(y))

    # RMSNorm: cast fp16 -> fp32, normalize, round to fp16, multiply by weight
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rrms = tl.math.rsqrt(ms + eps_rms)
    z = (yf * rrms).to(tl.float16)

    w = tl.load(RMS_W + cols, mask=mask, other=0.0)
    out = (z.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS half GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
