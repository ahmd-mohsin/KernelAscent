import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 555
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _rms_gelu_relu_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    xn = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    xn = xn * w  # fp16 multiply (matches reference)

    g = xn.to(tl.float32)
    # exact (erf-based) GELU computed in fp32, matching PyTorch's half GELU opmath
    gelu = g * 0.5 * (1.0 + tl.math.erf(g * 0.7071067811865476))
    out = tl.maximum(gelu, 0.0).to(tl.float16)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_gelu_relu_kernel[(m,)](
            x, self.rms1_w, y, n, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
