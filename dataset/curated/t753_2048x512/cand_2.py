import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 753
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_gelu_ln_bias_kernel(
    X, G, B, B3, Out,
    stride_xm,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), matches F.gelu default
    inv_sqrt2 = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

    # layernorm over the row
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # match reference: layernorm result rounded to bf16, then bias added in bf16
    y = y.to(tl.bfloat16).to(tl.float32) + b3

    tl.store(Out + row * stride_xm + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core matmul
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_gelu_ln_bias_kernel[(m,)](
            h, self.ln2_g, self.ln2_b, self.b3, out,
            h.stride(0),
            N=n, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
