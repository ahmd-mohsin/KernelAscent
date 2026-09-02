import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 392
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_rms_rms_softmax_ln_kernel(
    X_ptr, Out_ptr,
    W1_ptr, W2_ptr, G_ptr, B_ptr,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 1 (compute fp32, round to fp16, then half*half mul in fp32 opmath) ----
    ms = tl.sum(x * x, axis=0) / N
    x = x * tl.math.rsqrt(ms + 1e-6)
    x = x.to(tl.float16).to(tl.float32)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w1).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms = tl.sum(x * x, axis=0) / N
    x = x * tl.math.rsqrt(ms + 1e-6)
    x = x.to(tl.float16).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 accumulate, fp16 output) ----
    xm = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.exp(x - xm)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- LayerNorm (fp32 compute) ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # ---- scale (fp32 opmath) ----
    y = (y * 1.0194).to(tl.float16)

    tl.store(Out_ptr + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_rms_softmax_ln_kernel[(m,)](
            x, out,
            self.rms1_w, self.rms2_w, self.ln4_g, self.ln4_b,
            x.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
