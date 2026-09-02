import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 971
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_gelu_relu_bias_softmax(
    X, B, Out,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), rounded to bf16 to match PyTorch intermediate
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # ReLU
    g = tl.maximum(g, tl.zeros_like(g))

    # bias add in fp32, round to bf16 (matches bf16 + bf16 semantics)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g.to(tl.float32) + b).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch's internal upcast)
    yf = tl.where(mask, y.to(tl.float32), float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_relu_bias_softmax[(m,)](
            h, self.b3, out,
            h.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
