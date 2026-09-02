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
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N, stride,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load row (matmul output, bf16) ----
    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, matching torch's bf16 softmax) ----
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.bfloat16)

    # ---- scalar scales (each op rounds back to bf16 like eager mode) ----
    p = (p.to(tl.float32) * 1.3463).to(tl.bfloat16)
    p = (p.to(tl.float32) * 1.0016).to(tl.bfloat16)

    # ---- layer norm (fp32 accumulation) ----
    xf = p.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((xf - mean) * inv_std * g + b).to(tl.bfloat16)

    # ---- RMS norm (explicit fp32, matching reference) ----
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    z = (yf * r).to(tl.bfloat16)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (z.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride + offs, out, mask=mask)


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
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_ln_rms_kernel[(Mrows,)](
            y, self.ln4_g, self.ln4_b, self.rms5_w, out,
            N, y.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
