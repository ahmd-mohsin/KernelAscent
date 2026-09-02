import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 952
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_relu_bias_softmax(
    X, B, Out,
    N,
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # GELU (exact, erf-based) computed in fp32, cast back to fp16 (matches PyTorch half gelu)
    xf = x.to(tl.float32)
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * inv_sqrt2))
    # ReLU
    g = tl.maximum(g, 0.0)
    # cast to fp16, then add bias in fp16 semantics (fp32 add of two fp16 is exact,
    # cast to fp16 gives the correctly rounded fp16 add)
    g16 = g.to(tl.float16)
    y16 = (g16.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # Softmax in fp32 (matches PyTorch half softmax accumulation)
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    num = tl.exp(yf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = (num / denom).to(tl.float16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_gelu_relu_bias_softmax[(m,)](
            y, self.b3, out,
            n,
            y.stride(0),
            out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
