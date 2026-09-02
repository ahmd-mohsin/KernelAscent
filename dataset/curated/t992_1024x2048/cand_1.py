import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 992
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B, W, Out, stride_x, stride_o, N, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # relu (fp16 exact)
    x = tl.maximum(x, 0.0)

    # add bias: fp32 opmath, cast back to fp16 (matches PyTorch half add)
    xf = x.to(tl.float32) + b.to(tl.float32)
    xh = xf.to(tl.float16)

    # gelu (erf-based), fp32 opmath, cast back to fp16
    g = xh.to(tl.float32)
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    gh = g.to(tl.float16)

    # RMSNorm in fp32 on fp16 values
    gf = gh.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    normed = (gf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS tensor-core matmul
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            y, self.b2, self.rms4_w, out,
            y.stride(0), out.stride(0), n, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
