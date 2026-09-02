import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 898
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_softmax_gelu_ln(
    X, OUT, G, B,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, then round to fp16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # relu(relu(x)) is identity on softmax output (>= 0)

    # gelu (erf-based, computed in fp32 from fp16 input, rounded back to fp16)
    v = sm.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = v * 0.5 * (1.0 + tl.math.erf(v * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)
    g = tl.where(mask, g, 0.0)

    # layernorm in fp32
    mean = tl.sum(g, axis=0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * gamma + beta

    tl.store(OUT + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mr, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_gelu_ln[(Mr,)](
            h, out, self.ln5_g, self.ln5_b,
            N, h.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
