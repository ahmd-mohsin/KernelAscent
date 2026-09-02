import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 546
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_epilogue_softmax(X, B, Y, N, stride_x, stride_y,
                            S1: tl.constexpr, S2: tl.constexpr,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # elementwise ops, each rounded to fp16 to match reference semantics
    t = (x.to(tl.float32) * S1).to(tl.float16).to(tl.float32)
    t = tl.maximum(t, 0.0)
    t = (t + b.to(tl.float32)).to(tl.float16).to(tl.float32)
    t = (t * S2).to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation (matches PyTorch half softmax)
    t = tl.where(mask, t, float('-inf'))
    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM with fp32 accumulate
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue_softmax[(m,)](
            h, self.b3, y, n,
            h.stride(0), y.stride(0),
            S1=1.4852, S2=1.1826,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
