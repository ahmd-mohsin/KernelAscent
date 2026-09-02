import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 326
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_act_softmax(X, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)  # bf16
    # relu
    x = tl.maximum(x, 0.0)
    # emulate bf16 rounding after each scalar multiply (matches reference)
    t = (x.to(tl.float32) * 1.0573).to(tl.bfloat16)
    t = (t.to(tl.float32) * 1.4239).to(tl.bfloat16)
    # second relu (no-op mathematically but kept)
    t = tl.maximum(t, 0.0)
    # bias add in bf16
    b = tl.load(B + offs, mask=mask, other=0.0)
    s = (t.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (matches torch internal accumulation)
    z = tl.where(mask, s.to(tl.float32), float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM with fp32 accumulation
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_act_softmax[(Mrows,)](
            h, self.b5, y, N,
            h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
