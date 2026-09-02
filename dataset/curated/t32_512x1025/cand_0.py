import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 32
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_rows_kernel(
    X_ptr, LNG_ptr, LNB_ptr, B4_ptr, RMSW_ptr, OUT_ptr,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, fp16 output like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LNG_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LNB_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 math, fp16 output) ----
    y_m = tl.where(mask, y, float('-inf'))
    mmax = tl.max(y_m, axis=0)
    e = tl.where(mask, tl.exp(y - mmax), 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # ---- GELU (erf-based, fp32 math, fp16 output) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = 0.5 * sm * (1.0 + tl.math.erf(sm * INV_SQRT2))
    gel = gel.to(tl.float16).to(tl.float32)

    # ---- add bias (half add == fp32 add + round) ----
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (gel + b4).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (explicit fp32 as in reference) ----
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    zn = (z * r).to(tl.float16).to(tl.float32)
    w = tl.load(RMSW_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (zn * w).to(tl.float16)

    tl.store(OUT_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        rows, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_rows_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.b4, self.rms5_w, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
