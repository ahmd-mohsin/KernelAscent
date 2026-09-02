import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 885
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_softmax_scale_bias_softmax(
    X, B, Out,
    stride_xm,
    N,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # First softmax (fp32 math, like PyTorch half softmax with float accumulation)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    sm1 = (e1 / s1).to(tl.float16)  # round back to fp16 like the reference

    # scale (fp16 arithmetic like reference elementwise ops) + bias
    scale = tl.full([1], SCALE, tl.float16)
    b = tl.load(B + cols, mask=mask, other=0.0)
    t = sm1 * scale + b

    # Second softmax
    tf = tl.where(mask, t.to(tl.float32), float('-inf'))
    m2 = tl.max(tf, axis=0)
    e2 = tl.exp(tf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Out + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_scale_bias_softmax[(Mrows,)](
            y, self.b3, out,
            y.stride(0),
            N,
            SCALE=1.3127,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
