import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 128
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_rows_kernel(X_ptr, OUT_ptr, G_ptr, B_ptr, W_ptr,
                       N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    x = tl.load(X_ptr + base + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 compute, round to fp16) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- layer norm (eps=1e-5, fp32 compute, round to fp16) ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * inv * g + b).to(tl.float16).to(tl.float32)

    # ---- softmax 2 ----
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- RMS norm (eps=1e-6): cast to fp16 first, multiply by w in fp16 ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x16 = (x * r).to(tl.float16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float16)
    x = (x16 * w).to(tl.float32)

    # ---- softmax 3 ----
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(OUT_ptr + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fast cuBLAS GEMM (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _fused_rows_kernel[(M_,)](
            h, out, self.ln2_g, self.ln2_b, self.rms4_w,
            N=N_, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
