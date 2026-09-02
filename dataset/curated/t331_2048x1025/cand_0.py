import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 331
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W1_ptr, W3_ptr, W4_ptr, OUT_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # load matmul output row (fp16 -> fp32)
    xf = tl.load(X_ptr + base + offs).to(tl.float32)

    # ---- RMSNorm 1 (fp32 stats, cast to fp16, fp16 weight multiply) ----
    r = tl.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    xh = (xf * r).to(tl.float16) * tl.load(W1_ptr + offs)

    # ---- exact GELU (compute in fp32, cast back to fp16) ----
    xf = xh.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    xh = g.to(tl.float16)

    # ---- RMSNorm 3 ----
    xf = xh.to(tl.float32)
    r = tl.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    xh = (xf * r).to(tl.float16) * tl.load(W3_ptr + offs)

    # ---- RMSNorm 4 ----
    xf = xh.to(tl.float32)
    r = tl.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    xh = (xf * r).to(tl.float16) * tl.load(W4_ptr + offs)

    # ---- softmax (fp32 accumulate, fp16 output) ----
    xf = xh.to(tl.float32)
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(OUT_ptr + base + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (fp32 accumulate) — matches reference matmul
        y = x @ self.W0
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        _fused_post_kernel[(rows,)](
            y, self.rms1_w, self.rms3_w, self.rms4_w, out,
            N=N, BLOCK=N,
            num_warps=8,
        )
        return out
