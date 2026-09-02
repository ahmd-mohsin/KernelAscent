import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 490
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_gelu_ln_gelu_relu(
    X, G, B, Y,
    N, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact, erf) computed in fp32, rounded to fp16 to match torch op dtype
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # layernorm (fp32 accumulation)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # gelu again
    y = y * 0.5 * (1.0 + tl.math.erf(y * INV_SQRT2))
    # relu
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_row + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _fused_gelu_ln_gelu_relu[(M_,)](
            x, self.ln2_g, self.ln2_b, y,
            N_, x.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
