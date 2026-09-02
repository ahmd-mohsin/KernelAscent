import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 872
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _ln_kernel(X, G, B, Y, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = X + row * D
    y_row = Y + row * D

    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    _sumsq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < D
        x = tl.load(x_row + idx, mask=m, other=0.0).to(tl.float32)
        _sum += x
        _sumsq += x * x
    mean = tl.sum(_sum, axis=0) / D
    var = tl.sum(_sumsq, axis=0) / D - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < D
        x = tl.load(x_row + idx, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + idx, mask=m, other=0.0).to(tl.float32)
        b = tl.load(B + idx, mask=m, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * g + b
        tl.store(y_row + idx, y.to(tl.bfloat16), mask=m)


@triton.jit
def _rms_relu_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    m = idx < N
    x = tl.load(X + row * N + idx, mask=m, other=0.0).to(tl.float32)
    # x = x * 1.25 rounded to bf16 (matches reference bf16 multiply)
    x = (x * 1.25).to(tl.bfloat16).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rr = 1.0 / tl.sqrt(ms + eps)
    xn = (x * rr).to(tl.bfloat16)
    w = tl.load(W + idx, mask=m, other=0.0)
    out = xn * w
    out = tl.maximum(out, 0.0)
    tl.store(Y + row * N + idx, out, mask=m)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.reshape(-1, d).contiguous()
        rows = x2d.shape[0]

        ln_out = torch.empty_like(x2d)
        _ln_kernel[(rows,)](
            x2d, self.ln0_g, self.ln0_b, ln_out, d, 1e-5,
            BLOCK=1024, num_warps=8,
        )

        y = ln_out @ self.W1  # bf16 tensor-core matmul

        n = y.shape[-1]
        out = torch.empty_like(y)
        _rms_relu_kernel[(rows,)](
            y, self.rms3_w, out, n, 1e-6,
            BLOCK=2048, num_warps=8,
        )
        return out.reshape(*orig_shape[:-1], n)
