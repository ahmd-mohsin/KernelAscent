import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 128
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_epilogue(X, LNG, LNB, RMSW, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    x = tl.load(X + base + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 compute, fp16 output rounding) ----
    x = x - tl.max(x, axis=0)
    e = tl.where(mask, tl.exp(x), 0.0)
    x = e / tl.sum(e, axis=0)
    x = x.to(tl.float16).to(tl.float32)

    # ---- layer norm (eps = 1e-5, fp32 stats, fp16 output) ----
    n_f = N * 1.0
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LNG + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LNB + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- softmax 2 ----
    xm = tl.where(mask, x, float('-inf'))
    x = xm - tl.max(xm, axis=0)
    e = tl.where(mask, tl.exp(x), 0.0)
    x = e / tl.sum(e, axis=0)
    x = x.to(tl.float16).to(tl.float32)

    # ---- rms norm: fp32 compute, cast fp16, then fp16 multiply by weight ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / n_f
    x = x * (1.0 / tl.sqrt(ms + 1e-6))
    xh = x.to(tl.float16)
    w = tl.load(RMSW + offs, mask=mask, other=0.0)  # fp16
    xh = xh * w  # fp16 arithmetic, matching PyTorch half elementwise mul
    x = xh.to(tl.float32)

    # ---- softmax 3 ----
    xm = tl.where(mask, x, float('-inf'))
    x = xm - tl.max(xm, axis=0)
    e = tl.where(mask, tl.exp(x), 0.0)
    x = e / tl.sum(e, axis=0)

    tl.store(Y + base + offs, x.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS TensorCore GEMM (same as reference)
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue[(m,)](
            h, self.ln2_g, self.ln2_b, self.rms4_w, y, n,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
