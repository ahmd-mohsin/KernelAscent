import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 35
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _ln_rms_relu_kernel(
    X, G, B, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch opmath for bf16)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * inv_std * g + b

    # Cast to bf16 to match layer_norm output dtype, then back to fp32 for RMSNorm
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)

    # (yf * rrms).to(bf16) * w  -> compute mul in fp32 then round (matches eltwise bf16 mul)
    z_bf = (yf * rrms).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (z_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # ReLU
    zero = tl.zeros_like(out)
    out = tl.maximum(out, zero)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _ln_rms_relu_kernel[(m,)](
            x, self.ln1_g, self.ln1_b, self.rms2_w, y,
            x.stride(0), y.stride(0),
            N=n,
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
