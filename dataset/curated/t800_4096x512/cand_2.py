import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 800
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _bias_rmsnorm_kernel(
    Y, B, W, OUT,
    stride_ym, stride_om,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride_ym + cols, mask=mask, other=0.0)  # bf16
    b = tl.load(B + cols, mask=mask, other=0.0)                    # bf16

    # bias add in bf16 to match reference (x + b1 in bf16)
    x = (y + b).to(tl.bfloat16)

    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    normed = (xf * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)  # bf16
    out = normed * w  # bf16 multiply to match reference

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _bias_rmsnorm_kernel[(m,)](
            y, self.b1, self.rms2_w, out,
            y.stride(0), out.stride(0),
            N=n, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
