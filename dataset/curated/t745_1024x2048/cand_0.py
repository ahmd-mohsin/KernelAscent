import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 745
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _gelu_ln_kernel(
    X, OUT, G, B,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), computed in fp32 then rounded to fp16 to match PyTorch
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)  # match half-precision intermediate

    # LayerNorm with fp32 accumulation
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y - mean) * rstd * g + b

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _gelu_ln_kernel[(Mrows,)](
            h, out, self.ln2_g, self.ln2_b,
            N, h.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
