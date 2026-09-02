import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 443
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_ln_softmax2_rms_kernel(
    X, G, B, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, like PyTorch for bf16) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    # cast back to bf16 (as reference materializes bf16 between ops)
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax #1 (fp32 accumulation) ----
    y_masked = tl.where(mask, y, float('-inf'))
    m = tl.max(y_masked, axis=0)
    e = tl.math.exp(y_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- Softmax #2 (fp32 accumulation) ----
    y_masked = tl.where(mask, y, float('-inf'))
    m = tl.max(y_masked, axis=0)
    e = tl.math.exp(y_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (explicit fp32, matching reference) ----
    ms = tl.sum(y * y, axis=0) / N
    rr = 1.0 / tl.sqrt(ms + EPS_RMS)
    yn = (y * rr).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (yn * w).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS matmul (already optimal on A100 tensor cores)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax2_rms_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.rms4_w, out,
            h.stride(0), out.stride(0),
            N=N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
