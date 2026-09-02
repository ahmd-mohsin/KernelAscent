import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 601
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_rows_kernel(
    X, OUT,
    G2, B2, G4, B4,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ReLU (result stored as bf16 in reference; relu is exact so no rounding change)
    x = tl.maximum(x, 0.0)

    n_f = N * 1.0

    # ---- LayerNorm 1 (fp32 math, bf16 round like PyTorch) ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)  # match intermediate bf16 storage

    # ---- Softmax (fp32 math, bf16 round) ----
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(tl.where(mask, p, 0.0), axis=0) / n_f
    d2 = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g4 + b4
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- Scale ----
    out = z * SCALE
    tl.store(OUT + row * stride_o + cols, out.to(OUT.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS bf16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_rows_kernel[(Mrows,)](
            h, out,
            self.ln2_g, self.ln2_b, self.ln4_g, self.ln4_b,
            N, h.stride(0), out.stride(0),
            EPS=1e-5,
            SCALE=1.2373,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
