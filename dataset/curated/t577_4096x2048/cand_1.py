import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 577
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_softmax_ln_rms(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 math, bf16 output like torch) ----
    mmax = tl.max(x, 0)
    e = tl.exp(x - mmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = (e / s).to(tl.bfloat16)

    # ---- two scalar multiplies (fp32 opmath, bf16 rounding each step) ----
    p = (p.to(tl.float32) * 1.3463).to(tl.bfloat16)
    p = (p.to(tl.float32) * 1.0016).to(tl.bfloat16)

    # ---- layer norm (fp32 math, bf16 output) ----
    xf = p.to(tl.float32)
    mean = tl.sum(xf, 0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g + b).to(tl.bfloat16)

    # ---- rms norm (fp32 math, bf16 cast, then bf16*bf16 with fp32 opmath) ----
    yf = y.to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(yf * yf, 0) / N + 1e-6)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((yf * r).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]

        y = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_softmax_ln_rms[(rows,)](
            h2, self.ln4_g, self.ln4_b, self.rms5_w, y,
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
