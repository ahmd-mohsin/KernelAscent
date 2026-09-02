import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 321
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W, B, Y,
    D_size,
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_size

    # ---- load (bf16 -> fp32, exact) ----
    x = tl.load(X + row * stride_xm + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax in fp32 (matches torch's opmath for bf16) ----
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # round softmax result to bf16 (torch returns bf16 output), then upcast
    s_bf = s.to(tl.bfloat16)
    sf = s_bf.to(tl.float32)

    # ---- RMSNorm: rsqrt(mean(x^2) + eps) computed in fp32 ----
    sq = tl.where(mask, sf * sf, 0.0)
    ms = tl.sum(sq, axis=0) / D_size
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # (xf * rsqrt).to(bf16) * w  -- product of two bf16 is exact in fp32,
    # single rounding to bf16 matches bf16 multiply semantics
    n_bf = (sf * inv).to(tl.bfloat16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    h = (n_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # ---- exact GELU in fp32 (torch opmath), round to bf16 ----
    hf = h.to(tl.float32)
    g = hf * 0.5 * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    g_bf = g.to(tl.bfloat16)

    # ---- add bias (bf16 add: exact in fp32, single rounding) ----
    b = tl.load(B + offs, mask=mask, other=0.0)
    out = (g_bf.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.rms1_w, self.b3, y,
            d,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
