import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 749
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (compute in fp32, round to fp16 like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + RMS_EPS)
    xh = (xf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    xh = (xh.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # GELU (erf-based, fp32 math, fp16 storage rounding)
    g32 = xh.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = 0.5 * g32 * (1.0 + tl.math.erf(g32 * INV_SQRT2))
    xh = gel.to(tl.float16)

    # ReLU
    xh = tl.maximum(xh, 0.0)

    # scale (fp32 math, fp16 rounding)
    xh = (xh.to(tl.float32) * 1.2045).to(tl.float16)

    # LayerNorm (fp32 accumulation)
    v = xh.to(tl.float32)
    v = tl.where(mask, v, 0.0)
    mean = tl.sum(v, axis=0) / N
    d = tl.where(mask, v - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    gamma = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (v - mean) * rstd * gamma + beta

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_post_kernel[(m,)](
            x, self.rms1_w, self.ln5_g, self.ln5_b, y,
            n, x.stride(0), y.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
