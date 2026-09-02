import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 65
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_act_ln_kernel(
    X_ptr, B_ptr, G_ptr, Beta_ptr, Y_ptr,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # relu (fp16 -> fp16, exact)
    x = tl.maximum(x, 0.0)

    # gelu (computed in fp32, rounded to fp16 like PyTorch op output)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # + bias (opmath fp32, output rounded to fp16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    x = x.to(tl.float16).to(tl.float32)

    # gelu again
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # layer norm in fp32
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + beta

    tl.store(Y_ptr + row * N + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_act_ln_kernel[(Mrows,)](
            x, self.b3, self.ln5_g, self.ln5_b, y,
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
