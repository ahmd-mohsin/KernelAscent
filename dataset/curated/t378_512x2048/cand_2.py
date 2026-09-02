import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 378
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_rms_relu_ln_kernel(
    X, W_RMS, G, B, Y,
    N, stride_x, stride_y,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, rounded to bf16, then bf16 multiply by weight)
    ms = tl.sum(xf * xf, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(ms + RMS_EPS)
    xn = (xf * inv_rms).to(tl.bfloat16)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0).to(tl.bfloat16)
    h = (xn * w).to(tl.bfloat16)

    # ReLU
    h = tl.maximum(h, tl.zeros_like(h))

    # LayerNorm in fp32
    hf = h.to(tl.float32)
    mean = tl.sum(hf, axis=0) / N
    d = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_rms_relu_ln_kernel[(Mrows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, y,
            N, x2.stride(0), y.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
