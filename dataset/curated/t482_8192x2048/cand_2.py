import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 482
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_softmax_gelu(X, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)                   # fp16

    # emulate fp16 elementwise: (x * 1.024) + b, rounded to fp16 at each step
    scale = tl.full((1,), 1.024, dtype=tl.float16)
    t = (x * scale).to(tl.float16)
    t = (t + b).to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax which upcasts)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    m = tl.max(tf, axis=0)
    e = tl.exp(tf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # cast softmax result to fp16 (softmax output dtype), then exact GELU in fp32
    z = sm.to(tl.float16).to(tl.float32)
    g = z * 0.5 * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_gelu[(m,)](
            h, self.b2, out, n,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
