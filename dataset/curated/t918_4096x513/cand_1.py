import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 918
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_rms_gelu_kernel(
    Y, B2, W, OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.1963 (fp16 rounding after float compute, matching PyTorch opmath)
    a = (y * 1.1963).to(tl.float16).to(tl.float32)
    # x = x + b2
    b = (a + b2).to(tl.float16).to(tl.float32)

    # RMSNorm in float32
    ss = tl.sum(tl.where(mask, b * b, 0.0), axis=0)
    mean = ss / N
    r = tl.math.rsqrt(mean + 1e-6)

    c = (b * r).to(tl.float16).to(tl.float32)
    d = (c * w).to(tl.float16).to(tl.float32)

    # exact GELU (erf-based) computed in float, cast to fp16
    g = 0.5 * d * (1.0 + tl.math.erf(d * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    out = (g * 1.042).to(tl.float16)
    tl.store(OUT + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS GEMM (fp16, tensor cores)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_gelu_kernel[(m,)](
            y, self.b2, self.rms3_w, out,
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
