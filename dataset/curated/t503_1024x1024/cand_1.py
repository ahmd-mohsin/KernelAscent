import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 503
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, RMS_W, B2, G3, B3, G5, B5, OUT,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    x = x16.to(tl.float32)

    # RMSNorm (fp32 reduction), cast to fp16, multiply by weight in fp16
    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (x * rrms).to(tl.float16)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0)
    y16 = y16 * w

    # add bias in fp16
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    y16 = y16 + b2

    # LayerNorm 1 (fp32 compute, fp16 output)
    y = y16.to(tl.float32)
    mean = tl.sum(y, axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z16 = (d * rstd * g3 + b3).to(tl.float16)

    # GELU (erf, fp32 compute, fp16 output)
    z = z16.to(tl.float32)
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # LayerNorm 2 (fp32 compute, fp16 output)
    u = g16.to(tl.float32)
    mean2 = tl.sum(tl.where(mask, u, 0.0), axis=0) / N
    d2 = tl.where(mask, u - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g5 = tl.load(G5 + cols, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + cols, mask=mask, other=0.0).to(tl.float32)
    out16 = (d2 * rstd2 * g5 + b5).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(Mrows,)](
            h, self.rms1_w, self.b2,
            self.ln3_g, self.ln3_b, self.ln5_g, self.ln5_b,
            out, h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
