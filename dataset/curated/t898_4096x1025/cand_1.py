import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 898
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_softmax_gelu_ln_kernel(
    X, OUT, G, B,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch half softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom
    # round to fp16 (PyTorch stores softmax output in fp16)
    s = s.to(tl.float16)

    # relu twice: no-op on non-negative values (exact identity)

    # gelu (erf-based), computed in fp32 on fp16 inputs, rounded to fp16
    sf = s.to(tl.float32)
    g = 0.5 * sf * (1.0 + tl.math.erf(sf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # layer norm: stats in fp32 over fp16 values
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)
    mean = tl.sum(gf, axis=0) / N
    diff = tl.where(mask, gf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (gf - mean) * rstd * gamma + beta
    tl.store(OUT + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 matmul
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_softmax_gelu_ln_kernel[(m,)](
            y, out, self.ln5_g, self.ln5_b,
            y.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
