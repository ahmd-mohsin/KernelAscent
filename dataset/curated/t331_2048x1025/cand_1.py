import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 331
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_kernel(X, W1, W3, W4, OUT,
                  N, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x16.to(tl.float32)

    # RMSNorm 1
    r1 = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    y = (xf * r1).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w1).to(tl.float16)

    # GELU (exact, erf-based, computed in fp32 like PyTorch opmath)
    gf = y.to(tl.float32)
    gf = 0.5 * gf * (1.0 + tl.math.erf(gf * 0.7071067811865476))
    g16 = gf.to(tl.float16)

    # RMSNorm 3
    xf = g16.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    r3 = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    y = (xf * r3).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w3).to(tl.float16)

    # RMSNorm 4
    xf = y.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    r4 = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    y = (xf * r4).to(tl.float16)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w4).to(tl.float16)

    # Softmax (fp32 accumulation, like PyTorch on fp16 input)
    sf = y.to(tl.float32)
    sf = tl.where(mask, sf, float('-inf'))
    m = tl.max(sf, axis=0)
    e = tl.exp(sf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.rms1_w, self.rms3_w, self.rms4_w, out,
            N, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
