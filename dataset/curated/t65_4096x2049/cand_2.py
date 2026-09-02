import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 65
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, B_ptr, G_ptr, Beta_ptr, Out_ptr,
    N, stride_row,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # gelu (exact, erf)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    # + bias
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    # gelu again
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # layernorm
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = (x - mean) * rstd

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xn * g + beta

    tl.store(Out_ptr + row * stride_row + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        y = x @ self.W0
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue_kernel[(rows,)](
            y, self.b3, self.ln5_g, self.ln5_b, out,
            N, y.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
