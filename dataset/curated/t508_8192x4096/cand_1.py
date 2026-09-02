import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 508
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_gelu_ln_bias(
    X, G, B, B4, Out,
    stride,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    ptr = X + row * stride + offs
    x = tl.load(ptr, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulate, matches PyTorch half softmax)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)  # round to fp16 like reference

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)  # round to fp16 like reference
    g = tl.where(mask, g, 0.0)

    # layer norm (fp32 accumulate, biased variance, eps=1e-5)
    mean = tl.sum(g, 0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = tl.math.rsqrt(var + 1e-5)

    gamma = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * gamma + beta
    y16 = y.to(tl.float16)

    # + b4 in fp16 like reference
    b4 = tl.load(B4 + offs, mask=mask, other=0.0)
    out = y16 + b4

    tl.store(Out + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = x @ self.W0  # (M, 512), fp16, contiguous

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_gelu_ln_bias[(Mrows,)](
            h, self.ln3_g, self.ln3_b, self.b4, out,
            h.stride(0),
            N=N,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
