import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 729
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _gelu_bias_softmax_kernel(
    X, B, Out,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf) in fp32, round to fp16 to match reference step
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # scale, round to fp16
    g = (g * 1.3162).to(tl.float16).to(tl.float32)

    # bias add, round to fp16
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    g = (g + b).to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch fp16 softmax which upcasts)
    g = tl.where(mask, g, float("-inf"))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores)
        y = x @ self.W0
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _gelu_bias_softmax_kernel[(Mrows,)](
            y, self.b3, out,
            N, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8 if BLOCK_N >= 512 else 4,
        )
        return out
