import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 516
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_softmax_ln_gelu(
    X, G, B, Out,
    N, stride_x, stride_o, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, matching PyTorch half softmax) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y16 = (e / s).to(tl.float16)          # round to fp16 like reference output

    # ---- relu (no-op numerically for softmax output, kept for equivalence) ----
    y16 = tl.maximum(y16, 0.0)
    y = y16.to(tl.float32)

    # ---- layer norm (fp32 stats from fp16 input, like PyTorch) ----
    y = tl.where(mask, y, 0.0)
    mean = tl.sum(y, axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    z16 = (d * rstd * g + b).to(tl.float16)   # ln output rounded to fp16
    z = z16.to(tl.float32)

    # ---- exact (erf) GELU in fp32 ----
    out = z * 0.5 * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Out + row * stride_o + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_ln_gelu[(m,)](
            h, self.ln3_g, self.ln3_b, out,
            n, h.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
