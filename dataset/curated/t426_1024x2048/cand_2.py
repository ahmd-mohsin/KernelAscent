import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 426
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    stride_xm, stride_ym,
    N: tl.constexpr,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch bf16 layer_norm upcast)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # Cast to bf16 (as reference does), then back to fp32 for RMSNorm
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)

    ms = tl.sum(yf * yf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)

    t = (yf * rrms).to(tl.bfloat16)  # matches .to(x.dtype)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    # PyTorch bf16*bf16 elementwise: fp32 opmath, single rounding to bf16
    out = (t.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        y = torch.empty_like(h)
        grid = (Mrows,)
        _fused_ln_rms_kernel[grid](
            h, self.ln1_g, self.ln1_b, self.rms2_w, y,
            h.stride(0), y.stride(0),
            N=N,
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            BLOCK=512,
            num_warps=4,
        )
        return y
