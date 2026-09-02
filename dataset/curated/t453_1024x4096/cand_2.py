import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 453
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_relu_ln_bias_gelu2_kernel(
    X_ptr, G_ptr, B_ptr, B3_ptr, Y_ptr,
    N, stride_x, stride_y,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    # relu (bf16, exact)
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # layer norm (fp32 compute, two-pass)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b
    # round to bf16 (op boundary)
    y = y.to(tl.bfloat16)

    # add b3 (fp32 opmath, round to bf16)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) + b3).to(tl.bfloat16)

    # gelu (erf), fp32 opmath, round to bf16
    yf = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    yf = yf * 0.5 * (1.0 + tl.math.erf(yf * INV_SQRT2))
    y = yf.to(tl.bfloat16)

    # second gelu
    yf = y.to(tl.float32)
    yf = yf * 0.5 * (1.0 + tl.math.erf(yf * INV_SQRT2))
    y = yf.to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(N)
        _fused_relu_ln_bias_gelu2_kernel[(Mrows,)](
            x, self.ln2_g, self.ln2_b, self.b3, y,
            N, x.stride(0), y.stride(0),
            1e-5,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
        return y
