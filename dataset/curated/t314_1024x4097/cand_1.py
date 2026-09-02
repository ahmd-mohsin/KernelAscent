import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 314
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_act_softmax(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2 = 0.7071067811865476

    # gelu (exact, computed in fp32, rounded to fp16 like PyTorch's half kernel)
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # gelu again
    g2 = g * 0.5 * (1.0 + tl.math.erf(g * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)

    # scale
    v = g2 * 1.2863
    v = v.to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch's half softmax)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, 0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (fp16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_act_softmax[(rows,)](
            h, out, N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
