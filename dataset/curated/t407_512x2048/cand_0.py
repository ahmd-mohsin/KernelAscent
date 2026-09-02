import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 407
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B0, G, B, Y,
    N, D,
    stride_xm, stride_ym,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0 (stored as bf16 in reference)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # gelu (exact, erf)
    inv_sqrt2 = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x - mean) * rstd * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # softmax
    xmax = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    x = e / denom
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

    tl.store(Y + row * stride_ym + cols, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(n,)](
            x2, self.b0, self.ln2_g, self.ln2_b, y,
            n, d,
            x2.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
