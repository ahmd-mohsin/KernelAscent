import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 791
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)  # fp16

    # replicate the two sequential fp16 multiplies exactly
    c1 = tl.full((), 1.3036, tl.float16)
    c2 = tl.full((), 1.0726, tl.float16)
    x = x * c1
    x = x * c2

    # GELU (exact erf) computed in fp32 (matches PyTorch opmath for half), cast back to fp16
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    gh = g.to(tl.float16)

    # Softmax with fp32 accumulation (matches PyTorch half softmax)
    gf = gh.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    row_max = tl.max(gf, axis=0)
    e = tl.exp(gf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_gelu_softmax[(Mrows,)](h, y, N, BLOCK=BLOCK, num_warps=4)
        return y
