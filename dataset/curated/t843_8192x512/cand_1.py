import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 843
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu_ln_kernel(
    X, G, B, Y,
    N, eps,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # x = x * 1.0746  (bf16 storage rounding to match eager)
    t = (x.to(tl.float32) * 1.0746).to(tl.bfloat16)

    # exact gelu (erf), computed in fp32, rounded to bf16
    tf = t.to(tl.float32)
    g = tf * 0.5 * (1.0 + tl.math.erf(tf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # x = x * 1.326
    u = (g.to(tl.float32) * 1.326).to(tl.bfloat16)

    # layer norm in fp32
    uf = tl.where(mask, u.to(tl.float32), 0.0)
    mean = tl.sum(uf, axis=0) / N
    d = tl.where(mask, uf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * gamma + beta
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_kernel[(Mrows,)](
            h, self.ln4_g, self.ln4_b, y,
            N, 1e-5,
            h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
