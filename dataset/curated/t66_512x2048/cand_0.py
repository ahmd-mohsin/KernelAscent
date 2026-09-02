import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 66
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, OUT, W1, W3, G4, B4,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_h = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x_h.to(tl.float32)

    # RMSNorm 1 (compute in fp32, cast to fp16, then weight mul in fp32, cast fp16)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn_h = (xf * inv).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    x_h = (xn_h.to(tl.float32) * w1).to(tl.float16)

    # GELU (exact, erf), opmath fp32, cast fp16
    xf = x_h.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x_h = g.to(tl.float16)

    # RMSNorm 3
    xf = x_h.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn_h = (xf * inv).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    x_h = (xn_h.to(tl.float32) * w3).to(tl.float16)

    # LayerNorm (fp32 stats, eps 1e-5), cast fp16
    xf = x_h.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y_h = ((xf - mean) * rstd * g4 + b4).to(tl.float16)

    # final scale in fp32, cast fp16
    out = (y_h.to(tl.float32) * SCALE).to(tl.float16)
    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_norm_kernel[(m,)](
            h, out, self.rms1_w, self.rms3_w, self.ln4_g, self.ln4_b,
            n, h.stride(0), out.stride(0),
            SCALE=1.2037, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
