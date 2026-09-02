import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 960
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_softmax_rms_gelu_softmax(
    x_ptr, w_ptr, out_ptr,
    D, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- load row (fp16 -> fp32) ----
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask,
                other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 compute, like PyTorch half softmax) ----
    m1 = tl.max(x, axis=0)
    e1 = tl.math.exp(x - m1)          # masked lanes: exp(-inf) = 0
    s1 = tl.sum(e1, axis=0)
    p = e1 / s1
    p16 = p.to(tl.float16)            # stored as fp16 in reference

    # ---- RMSNorm: xf = p16.float(); xf * rsqrt(mean(xf^2)+eps) -> fp16, * w ----
    xf = p16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    t16 = (xf * r).to(tl.float16)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)   # fp16
    t16 = t16 * w                     # fp16 multiply (matches half*half)

    # ---- GELU (exact erf, fp32 opmath like PyTorch CUDA half gelu) ----
    g = t16.to(tl.float32)
    g = g * 0.5 * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g16 = g.to(tl.float16)            # gelu output stored as fp16

    # ---- softmax #2 ----
    gf = tl.where(mask, g16.to(tl.float32), float('-inf'))
    m2 = tl.max(gf, axis=0)
    e2 = tl.math.exp(gf - m2)
    s2 = tl.sum(e2, axis=0)
    o = (e2 / s2).to(tl.float16)

    tl.store(out_ptr + row * stride_o + offs, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, D_)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(D_)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_softmax_rms_gelu_softmax[(x2.shape[0],)](
            x2, self.rms1_w, out,
            D_, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(x.shape)
