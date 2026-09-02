import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 775
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X_ptr, OUT_ptr,
    G_ptr, B_ptr, B4_ptr, RW_ptr,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, bf16 rounding after) ----
    n = N
    mean = tl.sum(x, axis=0) / n
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax 1 (fp32 math, bf16 rounding) ----
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax 2 ----
    p_m = tl.where(mask, p, float('-inf'))
    mx2 = tl.max(p_m, axis=0)
    e2 = tl.exp(p_m - mx2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2
    p2 = p2.to(tl.bfloat16).to(tl.float32)

    # ---- Add bias (bf16 rounding) ----
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (p2 + b4).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32, bf16 round, then bf16 mul with weight) ----
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / n
    inv = 1.0 / tl.sqrt(ms + RMS_EPS)
    zn = (z * inv).to(tl.bfloat16).to(tl.float32)
    rw = tl.load(RW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (zn * rw).to(tl.bfloat16)

    tl.store(OUT_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_row_kernel[(m,)](
            h, out,
            self.ln1_g, self.ln1_b, self.b4, self.rms5_w,
            n, h.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
