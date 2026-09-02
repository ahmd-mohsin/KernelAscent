import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 914
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X, W_RMS, LN_G, LN_B, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (fp32 accumulate, then cast back to fp16 like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    y_h = (xf * rrms).to(tl.float16)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    y_h = y_h * w  # fp16 multiply, matching reference

    # LayerNorm in fp32 (matches PyTorch half layer_norm internals)
    yf = y_h.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)

    out = (yf - mean) * rstd * g + b
    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_rms_ln_kernel[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, out,
            x.stride(0), out.stride(0),
            N=n, EPS_RMS=1e-6, EPS_LN=1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
