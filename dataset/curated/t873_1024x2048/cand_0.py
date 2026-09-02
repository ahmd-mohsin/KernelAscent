import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 873
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _gelu_softmax_scale_relu_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    # mimic bf16 rounding of gelu output before softmax
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.where(mask, g, float('-inf'))

    # softmax in fp32
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to bf16 (softmax output dtype), then scale, then relu
    p = p.to(tl.bfloat16).to(tl.float32)
    out = p * SCALE
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _gelu_softmax_scale_relu_kernel[(Mrows,)](
            x, y, N,
            x.stride(0), y.stride(0),
            SCALE=1.1536,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
