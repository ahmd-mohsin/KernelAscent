import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 430
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    SQRT_HALF: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf-based), rounded to fp16 like reference intermediate
    g = 0.5 * x * (1.0 + tl.math.erf(x * SQRT_HALF))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32, output rounded to fp16
    g_m = tl.where(mask, g, float('-inf'))
    mx = tl.max(g_m, 0)
    e = tl.math.exp(g_m - mx)
    e = tl.where(mask, e, 0.0)
    s = e / tl.sum(e, 0)
    s = s.to(tl.float16).to(tl.float32)

    # second gelu, rounded to fp16
    g2 = 0.5 * s * (1.0 + tl.math.erf(s * SQRT_HALF))
    g2 = g2.to(tl.float16).to(tl.float32)

    # scale (fp32 opmath, round to fp16 like torch half*scalar)
    y = (g2 * 1.1055).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    normed = (y * r).to(tl.float16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (normed * w).to(tl.float16)

    tl.store(Y + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (fp16 tensor cores, fp32 accumulate)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            h, self.rms5_w, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
