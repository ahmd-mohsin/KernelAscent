import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 94
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_act_bias_softmax(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)

    # relu (bf16 -> same values)
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)

    # gelu (erf-based, computed in fp32 like torch opmath), round back to bf16
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # relu (no value change on nonneg, but keep semantics)
    gf = tl.maximum(g.to(tl.float32), 0.0)

    # bias add: torch does fp32 opmath then rounds to bf16
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (gf + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_act_bias_softmax[(Mrows,)](
            h, self.b4, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
