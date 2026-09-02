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
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)

    # x = x * 1.0746  (bf16 rounding to match reference)
    x = (x.to(tl.float32) * 1.0746).to(tl.bfloat16)

    # exact GELU in fp32, round to bf16 (matches F.gelu on bf16 tensor)
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = g.to(tl.bfloat16)

    # x = x * 1.326 (bf16 rounding)
    x = (x.to(tl.float32) * 1.326).to(tl.bfloat16)

    # LayerNorm in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xf - mean) * rstd * gamma + beta
    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        m, n = h.shape
        y = torch.empty_like(h)
        grid = (m,)
        _fused_gelu_ln_kernel[grid](
            h, self.ln4_g, self.ln4_b, y,
            n, h.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
