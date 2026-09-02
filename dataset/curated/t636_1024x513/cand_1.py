import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 636
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, B2_ptr, RW_ptr, LG_ptr, LB_ptr, OUT_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load matmul output row (bf16 -> fp32) ----
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accumulation, bf16 output like PyTorch) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    sm1 = (e1 / s1).to(tl.bfloat16)

    # ---- add bias in bf16 (matches x + b2 on bf16 tensors) ----
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0).to(tl.bfloat16)
    y = sm1 + b2

    # ---- RMSNorm: fp32 mean of squares, cast normalized to bf16, mul weight in bf16 ----
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), 0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    rw = tl.load(RW_ptr + offs, mask=mask, other=0.0).to(tl.bfloat16)
    z = (yf * rrms).to(tl.bfloat16) * rw

    # ---- LayerNorm (fp32 stats and affine, bf16 output) ----
    zf = z.to(tl.float32)
    mu = tl.sum(tl.where(mask, zf, 0.0), 0) / N
    d = tl.where(mask, zf - mu, 0.0)
    var = tl.sum(d * d, 0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LG_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(LB_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ln = ((zf - mu) * inv * g + bb).to(tl.bfloat16)

    # ---- softmax #2 ----
    lf = ln.to(tl.float32)
    lf_m = tl.where(mask, lf, float('-inf'))
    m2 = tl.max(lf_m, 0)
    e2 = tl.exp(lf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(OUT_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_post_kernel[(Mrows,)](
            h, self.b2, self.rms3_w, self.ln4_g, self.ln4_b, out,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
