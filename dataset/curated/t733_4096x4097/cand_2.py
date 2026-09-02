import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 733
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_norms_gelu_kernel(
    X, OUT,
    RMS2_W, LN3_G, LN3_B, RMS4_W,
    N,
    stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # scale (opmath float, cast back to fp16 as PyTorch does)
    xf = x.to(tl.float32) * SCALE
    x16 = xf.to(tl.float16)

    # RMSNorm 2
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * rinv).to(tl.float16)
    w2 = tl.load(RMS2_W + cols, mask=mask, other=0.0)
    y16 = (y16.to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    # LayerNorm 3 (computed in fp32 internally)
    xf = y16.to(tl.float32)
    mu = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mu, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN3_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN3_B + cols, mask=mask, other=0.0).to(tl.float32)
    z16 = (diff * rstd * g + b).to(tl.float16)

    # RMSNorm 4
    xf = z16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * rinv).to(tl.float16)
    w4 = tl.load(RMS4_W + cols, mask=mask, other=0.0)
    y16 = (y16.to(tl.float32) * w4.to(tl.float32)).to(tl.float16)

    # exact GELU (erf-based, fp32 opmath)
    xf = y16.to(tl.float32)
    out = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_norms_gelu_kernel[(m,)](
            h, out,
            self.rms2_w, self.ln3_g, self.ln3_b, self.rms4_w,
            n,
            h.stride(0), out.stride(0),
            SCALE=1.4626,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
