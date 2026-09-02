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

    # ---- softmax #1 (fp32 compute, round to fp16 like torch does) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    x = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- exact GELU (erf), fp32 compute then round to fp16 ----
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = g.to(tl.float16).to(tl.float32)

    # ---- scale, round to fp16 ----
    x = (x * 1.2534).to(tl.float16).to(tl.float32)

    # ---- softmax #2 ----
    x = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x, 0)
    e2 = tl.exp(x - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    y = e2 / s2

    # ---- relu (no-op on softmax output, kept for exactness) ----
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (fp16)
        h = x @ self.W0
        if not h.is_cuda:
            h = torch.softmax(h, dim=-1)
            h = F.gelu(h)
            h = h * 1.2534
            h = torch.softmax(h, dim=-1)
            return torch.relu(h)

        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_gelu_softmax[(rows,)](
            h, out, N, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
