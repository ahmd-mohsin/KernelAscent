import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 460
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_softmax_gelu_relu_ln(
    X, Y, G, B,
    N, eps,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, output rounded to bf16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # exact gelu in fp32 (PyTorch opmath), round to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu (exact on bf16 values)
    r = tl.maximum(g, 0.0)
    r = tl.where(mask, r, 0.0)

    # layernorm in fp32 (matches PyTorch bf16 layernorm internals)
    mean = tl.sum(r, axis=0) / N
    d = tl.where(mask, r - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (r - mean) * rstd * gamma + beta
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_gelu_relu_ln[(Mrows,)](
            h, out, self.ln4_g, self.ln4_b,
            N, 1e-5,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
