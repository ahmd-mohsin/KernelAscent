import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 344
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_gelu_scale_bias_softmax(
    X, B, Out,
    N, stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (erf-based), rounded to bf16 like the reference
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # x * 1.0846 (bf16 rounding)
    g = (g * 1.0846).to(tl.bfloat16).to(tl.float32)
    # x * 1.379 (bf16 rounding)
    g = (g * 1.379).to(tl.bfloat16).to(tl.float32)

    # + bias (bf16 rounding)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    g = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    g = tl.where(mask, g, float("-inf"))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out + row * stride_om + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (TF32/BF16 tensor cores)
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_scale_bias_softmax[(m,)](
            x, self.b4, out,
            n, x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
