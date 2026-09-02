import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 345
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _gelu_relu_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, computed in fp32 then rounded to bf16 to match
    # F.gelu on a bf16 tensor
    inv_sqrt2 = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    g = g.to(tl.bfloat16)

    # relu (exact on bf16)
    g = tl.maximum(g, 0.0)

    # softmax with fp32 accumulation (matches torch.softmax on bf16 input)
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    row_max = tl.max(gf, axis=0)
    e = tl.exp(gf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _gelu_relu_softmax_kernel[(m,)](
            x, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return y
