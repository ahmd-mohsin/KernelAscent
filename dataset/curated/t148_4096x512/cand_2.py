import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 148
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_scale_softmax_dualln(
    X, G3, B3, G4, B4, Y,
    stride_x, stride_y,
    N,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- load row (bf16 -> fp32) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- scale (round to bf16 to match reference intermediate) ----
    x = x * SCALE
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- softmax (fp32 accumulation, bf16 output rounding) ----
    row_max = tl.max(x, 0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, 0)
    p = e / denom
    p = p.to(tl.bfloat16).to(tl.float32)

    n_f = N.to(tl.float32)

    # ---- layer norm 1 ----
    mean1 = tl.sum(tl.where(mask, p, 0.0), 0) / n_f
    d1 = tl.where(mask, p - mean1, 0.0)
    var1 = tl.sum(d1 * d1, 0) / n_f
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g3 + b3
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- layer norm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), 0) / n_f
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g4 + b4

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (already optimal on A100)
        h = torch.matmul(x, self.W0)

        orig_shape = h.shape
        N = orig_shape[-1]
        h2d = h.reshape(-1, N)
        if not h2d.is_contiguous():
            h2d = h2d.contiguous()
        rows = h2d.shape[0]

        out = torch.empty_like(h2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_scale_softmax_dualln[(rows,)](
            h2d, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            h2d.stride(0), out.stride(0),
            N,
            SCALE=1.4775,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)
