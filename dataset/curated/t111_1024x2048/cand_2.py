import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 111
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _gelu_softmax_kernel(X, Y, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU (erf-based), rounded to bf16 to match F.gelu output dtype
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, 0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(Y + row * stride + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _ln_scale_kernel(X, G, B, Y, N, stride, eps, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # match: layer_norm produces bf16, then scalar multiply in bf16 arithmetic
    y = y.to(tl.bfloat16).to(tl.float32) * scale
    tl.store(Y + row * stride + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W2 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        # fused gelu + softmax
        y1 = torch.empty_like(x)
        BLOCK1 = triton.next_power_of_2(d)
        _gelu_softmax_kernel[(m,)](x, y1, d, x.stride(0), BLOCK=BLOCK1, num_warps=8)
        # matmul (tensor cores via cuBLAS)
        h = y1 @ self.W2
        # fused layernorm + scale
        n = h.shape[-1]
        out = torch.empty_like(h)
        BLOCK2 = triton.next_power_of_2(n)
        _ln_scale_kernel[(m,)](
            h, self.ln3_g, self.ln3_b, out, n, h.stride(0),
            1e-5, 1.4031, BLOCK=BLOCK2, num_warps=16,
        )
        return out
