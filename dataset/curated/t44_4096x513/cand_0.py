import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 44
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_bias_softmax_gelu_rms(
    X, B, W, OUT,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # bias add in fp16 (match reference rounding), then upcast
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    x = (x + b).to(tl.float32)

    # softmax in fp32, output rounded to fp16 (as torch does)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # exact GELU (erf-based), rounded to fp16
    pf = p.to(tl.float32)
    g = (0.5 * pf * (1.0 + tl.math.erf(pf * 0.7071067811865476))).to(tl.float16)

    # RMSNorm in fp32, cast to fp16, multiply by weight in fp16
    gf = g.to(tl.float32)
    ms = tl.sum(gf * gf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (gf * inv).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = y * w

    tl.store(OUT + row * stride_o + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_softmax_gelu_rms[(m,)](
            h, self.b1, self.rms4_w, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
