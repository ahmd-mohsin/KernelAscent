import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 936
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_gelu_relu_rmsnorm(Y, W, OUT, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU computed in fp32 (matches PyTorch half opmath),
    # then cast to fp16 as PyTorch would store the intermediate
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # relu in fp16
    r16 = tl.maximum(g16, 0.0)

    # RMS norm in fp32
    xf = r16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = tl.math.rsqrt(ms + eps)
    normed16 = (xf * rrms).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = normed16 * w
    tl.store(OUT + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_relu_rmsnorm[(m,)](
            y, self.rms3_w, out, n, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
