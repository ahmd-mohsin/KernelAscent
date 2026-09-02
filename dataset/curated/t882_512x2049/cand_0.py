import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 882
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X_ptr, W_ptr, Y_ptr,
    D: tl.constexpr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, erf-based), rounded to bf16 like PyTorch ----
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax 1 (fp32 accumulation, bf16 output) ----
    g_m = tl.where(mask, g, float('-inf'))
    m1 = tl.max(g_m, axis=0)
    e1 = tl.math.exp(g_m - m1)          # exp(-inf) = 0 handles masked lanes
    s1 = tl.sum(e1, axis=0)
    p = e1 / s1
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32) -> bf16, then * weight (bf16) ----
    ms = tl.sum(p * p, axis=0) / D
    r = p * tl.math.rsqrt(ms + 1e-6)
    r = r.to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = (r * w).to(tl.bfloat16).to(tl.float32)

    # ---- ReLU ----
    v = tl.maximum(v, 0.0)

    # ---- Softmax 2 ----
    v_m = tl.where(mask, v, float('-inf'))
    m2 = tl.max(v_m, axis=0)
    e2 = tl.math.exp(v_m - m2)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        w = self.rms2_w
        if not w.is_cuda:
            w = w.cuda()

        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(d)
        _fused_row_kernel[(rows,)](
            x, w, y,
            d,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
