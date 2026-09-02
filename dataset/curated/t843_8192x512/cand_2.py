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
    N,
    stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.0746  (bf16 rounding to match eager op-by-op behavior)
    x = (x * 1.0746).to(tl.bfloat16).to(tl.float32)

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))), computed in fp32, rounded to bf16
    x = (x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)

    # x = x * 1.326
    x = (x * 1.326).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xm * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_kernel[(Mrows,)](
            h, self.ln4_g, self.ln4_b, out,
            N,
            h.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
