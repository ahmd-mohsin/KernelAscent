import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 791
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # scale 1: fp32 math, round to fp16 (matches PyTorch half scalar mul)
    v = x.to(tl.float32) * 1.3036
    v = v.to(tl.float16)
    # scale 2
    v = v.to(tl.float32) * 1.0726
    v = v.to(tl.float16)

    # exact GELU (erf) computed in fp32, rounded to fp16 (matches F.gelu on half)
    vf = v.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * vf * (1.0 + tl.math.erf(vf * INV_SQRT2))
    g = g.to(tl.float16)

    # softmax in fp32 accumulation
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    m = tl.max(gf, axis=0)
    e = tl.exp(gf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_gelu_softmax[(Mrows,)](
            h, out, N, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return out
