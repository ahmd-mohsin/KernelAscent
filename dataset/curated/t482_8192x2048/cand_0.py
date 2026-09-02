import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 482
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_scale_bias_softmax_gelu(
    X, B, Y,
    N, stride_x, stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)                   # fp16

    # x * 1.024 : computed in fp32 (opmath), rounded to fp16
    t = (x.to(tl.float32) * 1.024).to(tl.float16)
    # x + b : computed in fp32 (opmath), rounded to fp16
    t = (t.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # softmax in fp32 (as PyTorch does for fp16 inputs)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    m = tl.max(tf, axis=0)
    e = tl.exp(tf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # exact GELU in fp32 opmath, rounded to fp16
    v = sm.to(tl.float32)
    g = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
    out = g.to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 matmul
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_scale_bias_softmax_gelu[(Mrows,)](
            h, self.b2, y,
            N, h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
