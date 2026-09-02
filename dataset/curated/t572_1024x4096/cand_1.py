import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 572
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, w_ptr, g_ptr, beta_ptr, out_ptr,
    N, stride_row,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)

    # x = x + b0 (bf16 add semantics: fp32 compute, round to bf16)
    y = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    # relu
    y = tl.maximum(y, 0.0)

    # RMSNorm in fp32
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = tl.math.rsqrt(ms + RMS_EPS)
    z = (yf * r).to(tl.bfloat16)

    # multiply by rms2_w (bf16 mul semantics)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm in fp32
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean = tl.sum(zf, axis=0) / N
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + LN_EPS)

    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd) * g + beta

    tl.store(out_ptr + row * stride_row + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x, self.b0, self.rms2_w, self.ln3_g, self.ln3_b, out,
            N, x.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
