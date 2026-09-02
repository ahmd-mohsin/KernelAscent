import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 577
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_softmax_ln_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    s1, s2, eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load row (bf16 -> f32) ----
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 math, round to bf16 like torch) ----
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = (e / denom).to(tl.bfloat16)

    # ---- two scalar multiplies (fp32 opmath, bf16 rounding each step, like torch) ----
    y = (y.to(tl.float32) * s1).to(tl.bfloat16)
    y = (y.to(tl.float32) * s2).to(tl.bfloat16)

    # ---- layer norm (fp32 accumulation, bf16 output, like torch F.layer_norm) ----
    xf = y.to(tl.float32)
    xf_m = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf_m, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.rsqrt(var + eps_ln)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ln = ((xf - mean) * rstd * g + b).to(tl.bfloat16)

    # ---- rms norm exactly as written in reference ----
    lf = ln.to(tl.float32)
    ms = tl.sum(tl.where(mask, lf * lf, 0.0), axis=0) / N
    r = (lf * tl.rsqrt(ms + eps_rms)).to(tl.bfloat16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (r.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_ln_rms_kernel[(Mrows,)](
            h, self.ln4_g, self.ln4_b, self.rms5_w, out,
            N, h.stride(0), out.stride(0),
            1.3463, 1.0016, 1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
