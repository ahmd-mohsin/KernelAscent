import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 4
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X, OUT, W2, G3, B3, B5,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # --- softmax (fp32 math, rounded to bf16 like torch.softmax output) ---
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16)

    # --- RMS norm in fp32 on the bf16 softmax values ---
    xf = p.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y_bf = (xf * r).to(tl.bfloat16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    y_bf = (y_bf.to(tl.float32) * w2).to(tl.bfloat16)

    # --- layer norm (fp32 stats) ---
    yf = y_bf.to(tl.float32)
    mean = tl.sum(tl.where(mask, yf, 0.0), axis=0) / N
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)

    g = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((yf - mean) * inv * g + b).to(tl.bfloat16)

    # --- scale ---
    z = (z.to(tl.float32) * 1.4134).to(tl.bfloat16)

    # --- bias add ---
    b5 = tl.load(B5 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) + b5).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # (M, 512) bf16 via tensor cores
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_post_kernel[(m,)](
            h, out, self.rms2_w, self.ln3_g, self.ln3_b, self.b5,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
