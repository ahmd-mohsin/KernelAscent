import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 909
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_ln_softmax_gelu(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch on half inputs)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    # round to fp16 (layer_norm output dtype) then back to fp32 for softmax
    y = y.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation, like PyTorch on half)
    y_masked = tl.where(mask, y, float('-inf'))
    m = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to fp16 (softmax output dtype) then gelu in fp32
    p = p.to(tl.float16).to(tl.float32)

    # exact (erf) GELU
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = p * 0.5 * (1.0 + tl.math.erf(p * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_softmax_gelu[(rows,)](
            h, self.ln1_g, self.ln1_b, out,
            h.stride(0), out.stride(0),
            N, 1e-5, BLOCK,
            num_warps=num_warps,
        )
        return out
