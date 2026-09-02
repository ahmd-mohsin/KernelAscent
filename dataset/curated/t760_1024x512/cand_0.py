import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 760
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_softmax_gelu_ln_kernel(
    X_ptr, G_ptr, B_ptr, Out_ptr,
    stride_row,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, matching PyTorch half softmax) ----
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom
    # round to fp16 (intermediate tensor materialization in reference)
    sm = sm.to(tl.float16).to(tl.float32)

    # ---- GELU (exact erf variant, fp32 opmath like PyTorch half gelu) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = sm * 0.5 * (1.0 + tl.math.erf(sm * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation) ----
    gm = tl.where(mask, g, 0.0)
    mean = tl.sum(gm, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = diff * rstd * w + b
    tl.store(Out_ptr + row * stride_row + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_gelu_ln_kernel[(m,)](
            h, self.ln3_g, self.ln3_b, out,
            h.stride(0),
            N=n,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
