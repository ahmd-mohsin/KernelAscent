import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 151
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, G2, B2, G3, B3,
    N, stride_x, stride_y,
    SCALE, EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (fp32 math, round to bf16 like PyTorch output) ----
    row_max = tl.max(x, axis=0)
    ex = tl.exp(x - row_max)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    sm = ex / denom
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # ---- scale, round to bf16 ----
    h = (sm * SCALE).to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 1 ----
    n_f = N.to(tl.float32)
    mean1 = tl.sum(tl.where(mask, h, 0.0), axis=0) / n_f
    d1 = tl.where(mask, h - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n_f
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = (d1 * rstd1) * g2 + b2
    y1 = y1.to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 2 ----
    mean2 = tl.sum(tl.where(mask, y1, 0.0), axis=0) / n_f
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = (d2 * rstd2) * g3 + b3

    tl.store(Y + row * stride_y + cols, y2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.reshape(-1, N)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2d, y,
            self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            N, x2d.stride(0), y.stride(0),
            1.4033, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
