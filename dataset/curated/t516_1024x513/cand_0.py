import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 516
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_softmax_ln_gelu(X, OUT, G, B, N, stride, eps,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, matches PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # relu is a no-op on softmax output (>= 0), but rounding to fp16 matches ref
    sm16 = sm.to(tl.float16)
    sf = sm16.to(tl.float32)

    # layernorm on fp16 values with fp32 accumulation
    mean = tl.sum(sf, axis=0) / N
    d = tl.where(mask, sf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (sf - mean) * rstd * g + b
    y16 = y.to(tl.float16)
    yf = y16.to(tl.float32)

    # exact GELU (erf)
    out = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(OUT + row * stride + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core matmul
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_ln_gelu[(m,)](
            h, out, self.ln3_g, self.ln3_b, n, h.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
