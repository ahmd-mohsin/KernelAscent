import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 878
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load + relu (fp32 math) ----
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # ---- softmax (fp32 accumulate, round to bf16 like PyTorch output) ----
    xm = tl.where(mask, x, float('-inf'))
    row_max = tl.max(xm, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom
    p = p.to(tl.bfloat16).to(tl.float32)  # match bf16 intermediate storage

    # ---- layer norm (fp32 stats, biased variance) ----
    mean = tl.sum(tl.where(mask, p, 0.0), axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)  # match bf16 intermediate storage

    # ---- exact GELU (erf-based, fp32 math) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.bfloat16).to(tl.float32)  # match bf16 intermediate storage

    # ---- scale ----
    y = y * scale

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_row_kernel[(Mrows,)](
            x, self.ln2_g, self.ln2_b, out,
            N, x.stride(0), out.stride(0),
            1e-5, 1.1802,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
