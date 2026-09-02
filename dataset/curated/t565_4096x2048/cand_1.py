import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 565
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_rowops_kernel(
    X_ptr, Out_ptr,
    ln1_g_ptr, ln1_b_ptr,
    rms2_w_ptr,
    ln5_g_ptr, ln5_b_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (eps=1e-5), fp32 math, output rounded to bf16 ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    inv1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g1 = tl.load(ln1_g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(ln1_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (d1 * inv1) * g1 + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (eps=1e-6): fp32 normalize -> bf16 -> * weight -> bf16 ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * rrms).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(rms2_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 math, bf16 output) ----
    xm = tl.where(mask, x, float("-inf"))
    m = tl.max(xm, axis=0)
    e = tl.exp(xm - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (erf-based, fp32 math, bf16 output) ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 5 (eps=1e-5) ----
    mean5 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d5 = tl.where(mask, x - mean5, 0.0)
    var5 = tl.sum(d5 * d5, axis=0) / N
    inv5 = 1.0 / tl.sqrt(var5 + 1e-5)
    g5 = tl.load(ln5_g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(ln5_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d5 * inv5) * g5 + b5

    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rowops_kernel[(Mrows,)](
            x, out,
            self.ln1_g, self.ln1_b,
            self.rms2_w,
            self.ln5_g, self.ln5_b,
            N, x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
