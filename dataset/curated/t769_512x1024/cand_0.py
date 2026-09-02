import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 769
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_epilogue_softmax(
    X, B, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # replicate fp16 rounding of each elementwise step
    t = (x * 1.2814).to(tl.float16)
    t = (t + b).to(tl.float16)
    t = (t * 1.0299).to(tl.float16)

    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    row_max = tl.max(tf, axis=0)
    e = tl.exp(tf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue_softmax[(m,)](
            h, self.b2, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
