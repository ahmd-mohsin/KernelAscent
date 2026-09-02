import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 597
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_softmax_gelu_softmax(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulate, round to fp16 like reference output dtype)
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    s = e / tl.sum(e, 0)
    s = s.to(tl.float16).to(tl.float32)

    # gelu (erf-based, default) then round to fp16
    g = 0.5 * s * (1.0 + tl.math.erf(s * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # scale, round to fp16
    g = g * 1.2534
    g = g.to(tl.float16).to(tl.float32)

    # softmax 2
    g = tl.where(mask, g, float('-inf'))
    g = g - tl.max(g, 0)
    e2 = tl.exp(g)
    s2 = e2 / tl.sum(e2, 0)

    # relu (softmax outputs are already >= 0, kept for exactness)
    s2 = tl.maximum(s2, 0.0)

    tl.store(Y + row * stride_y + offs, s2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_softmax_gelu_softmax[(Mrows,)](
            h, out, N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
