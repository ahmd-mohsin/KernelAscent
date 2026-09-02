import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 760
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _softmax_gelu_ln_kernel(
    X, Y, G, B,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32), round back to fp16 as PyTorch does between ops
    xmax = tl.max(x, axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # exact GELU (erf based), round back to fp16
    ge = 0.5 * sm * (1.0 + tl.math.erf(sm * 0.7071067811865476))
    ge = ge.to(tl.float16).to(tl.float32)

    # layernorm with fp32 stats
    mean = tl.sum(tl.where(mask, ge, 0.0), axis=0) / N
    d = tl.where(mask, ge - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y + row * stride_ym + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_gelu_ln_kernel[(m,)](
            h, y, self.ln3_g, self.ln3_b,
            h.stride(0), y.stride(0),
            n, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
